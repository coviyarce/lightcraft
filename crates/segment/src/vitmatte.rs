//! ViTMatte-S (Yao et al., HUST): alpha matting of an image from a trimap. A ViTDet backbone
//! ([`crate::vitdet`]) over RGB + trimap gives a stride-16 feature map; a small detail-capture
//! module (a strided convolution stream over the input, four upsample-concat-convolve fusion
//! blocks and a matting head) turns it into full-resolution alpha.
//!
//! Modified work (Apache License 2.0, §4(b)): ported by the LightCraft contributors in 2026 from
//! the Python/PyTorch ViTMatte code of Hugging Face Transformers (`models/vitmatte/modeling_vitmatte.py`
//! and `image_processing_vitmatte.py`), Copyright The HuggingFace Team; translated to Rust on
//! candle and restructured. See NOTICE.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use crate::vitdet::{HIDDEN, IN_CHANNELS, VitDet};
use crate::{Error, Result, weights};

/// The weights file of a ViTMatte-S model directory.
pub const MATTE_WEIGHTS_FILE: &str = "model.safetensors";
/// Input sizes are padded to a multiple of this.
pub const MATTE_DIVISOR: usize = 32;
/// Largest padded side accepted (global attention grows with the square of the token count).
pub const MATTE_MAX_SIDE: usize = 1024;

const CONVSTREAM: [usize; 3] = [48, 96, 192];
const FUSION: [usize; 4] = [256, 128, 64, 32];
const BN_EPS: f64 = 1e-5;

/// Whether `dir` holds a ViTMatte-S checkpoint.
pub fn is_matte_model_dir(dir: &Path) -> bool {
    dir.join(MATTE_WEIGHTS_FILE).is_file()
}

/// A convolution with its batch norm folded in (inference), followed by ReLU.
struct ConvBn {
    conv: Conv2d,
}

/// Shape of a [`ConvBn`]: input and output channels, kernel size, stride, and whether the
/// convolution has a bias of its own.
#[derive(Clone, Copy)]
struct ConvSpec {
    inp: usize,
    out: usize,
    k: usize,
    stride: usize,
    bias: bool,
}

impl ConvBn {
    /// `conv_vb` holds `weight` (and `bias` when the spec says so), `bn_vb` the batch norm's tensors.
    fn new(spec: ConvSpec, conv_vb: VarBuilder, bn_vb: VarBuilder) -> Result<Self> {
        let ConvSpec { inp, out, k, stride, bias } = spec;
        let w = conv_vb.get((out, inp, k, k), "weight")?;
        let b = if bias { conv_vb.get(out, "bias")? } else { Tensor::zeros(out, DType::F32, w.device())? };
        let (gamma, beta) = (bn_vb.get(out, "weight")?, bn_vb.get(out, "bias")?);
        let (mean, var) = (bn_vb.get(out, "running_mean")?, bn_vb.get(out, "running_var")?);
        let scale = (gamma / (var + BN_EPS)?.sqrt()?)?;
        let w = w.broadcast_mul(&scale.reshape((out, 1, 1, 1))?)?;
        let b = (((b - mean)? * &scale)? + beta)?;
        let cfg = Conv2dConfig { padding: k / 2, stride, ..Default::default() };
        Ok(Self { conv: Conv2d::new(w, Some(b), cfg) })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.conv.forward(x)?.relu()?)
    }
}

struct Decoder {
    convstream: Vec<ConvBn>,
    fusion: Vec<ConvBn>,
    head_conv: ConvBn,
    head_out: Conv2d,
}

impl Decoder {
    fn new(vb: VarBuilder) -> Result<Self> {
        let mut convstream = Vec::new();
        let chans = [IN_CHANNELS, CONVSTREAM[0], CONVSTREAM[1], CONVSTREAM[2]];
        for i in 0..CONVSTREAM.len() {
            let v = vb.pp(format!("convstream.convs.{i}"));
            convstream.push(ConvBn::new(
                ConvSpec { inp: chans[i], out: chans[i + 1], k: 3, stride: 2, bias: false },
                v.pp("conv"),
                v.pp("batch_norm"),
            )?);
        }
        let mut fusion = Vec::new();
        let ins = [HIDDEN, FUSION[0], FUSION[1], FUSION[2]];
        for i in 0..FUSION.len() {
            let detail = chans[chans.len() - 1 - i];
            let v = vb.pp(format!("fusion_blocks.{i}.conv"));
            fusion.push(ConvBn::new(
                ConvSpec { inp: ins[i] + detail, out: FUSION[i], k: 3, stride: 1, bias: false },
                v.pp("conv"),
                v.pp("batch_norm"),
            )?);
        }
        let h = vb.pp("matting_head.matting_convs");
        let head_conv = ConvBn::new(ConvSpec { inp: FUSION[3], out: 16, k: 3, stride: 1, bias: true }, h.pp("0"), h.pp("1"))?;
        let head_out = candle_nn::conv2d(16, 1, 1, Default::default(), h.pp("3"))?;
        Ok(Self { convstream, fusion, head_conv, head_out })
    }

    fn forward(&self, features: &Tensor, pixels: &Tensor, mut stages: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor> {
        let mut detail = vec![pixels.clone()];
        for c in &self.convstream {
            let next = c.forward(detail.last().unwrap_or(pixels))?;
            detail.push(next);
        }
        if let Some(s) = stages.as_deref_mut() {
            for (i, d) in detail.iter().enumerate().skip(1) {
                s.push((format!("detail_{i}"), d.clone()));
            }
        }
        let mut x = features.clone();
        for (i, f) in self.fusion.iter().enumerate() {
            let (_, _, h, w) = x.dims4()?;
            let up = x.upsample_bilinear2d(2 * h, 2 * w, false)?;
            let d = detail.get(detail.len() - 1 - i).ok_or_else(|| Error::Matte("detail stream is too short".into()))?;
            x = f.forward(&Tensor::cat(&[d, &up], 1)?)?;
            if let Some(s) = stages.as_deref_mut() {
                s.push((format!("fusion_{i}"), x.clone()));
            }
        }
        let a = candle_nn::ops::sigmoid(&self.head_out.forward(&self.head_conv.forward(&x)?)?)?;
        if let Some(s) = stages {
            s.push(("alphas".into(), a.clone()));
        }
        Ok(a)
    }
}

/// The planar `[4, H, W]` model input for an 8-bit RGB image and trimap (0 = background, 128 =
/// unknown, 255 = foreground), both `w × h` row-major: RGB in −1..1, the trimap in 0..1, zero padded
/// at the bottom and right to multiples of 32. Also returns `(H, W)`. Hostile sizes are errors.
fn prepare_input(rgb: &[u8], trimap: &[u8], w: usize, h: usize) -> Result<(Vec<f32>, (usize, usize))> {
    let n = w.checked_mul(h).filter(|n| *n > 0).ok_or_else(|| Error::Matte(format!("bad image size {w}x{h}")))?;
    if rgb.len() != n.saturating_mul(3) || trimap.len() != n {
        return Err(Error::Matte(format!(
            "image of {w}x{h} needs {} RGB bytes and {n} trimap bytes, got {} and {}",
            n.saturating_mul(3),
            rgb.len(),
            trimap.len()
        )));
    }
    let (hp, wp) = (h.div_ceil(MATTE_DIVISOR) * MATTE_DIVISOR, w.div_ceil(MATTE_DIVISOR) * MATTE_DIVISOR);
    if hp > MATTE_MAX_SIDE || wp > MATTE_MAX_SIDE {
        return Err(Error::Matte(format!("image of {w}x{h} is larger than the {MATTE_MAX_SIDE} px limit")));
    }
    let plane = hp * wp;
    let mut data = vec![0f32; IN_CHANNELS * plane];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let o = y * wp + x;
            for c in 0..3 {
                let v = rgb.get(3 * i + c).copied().unwrap_or(0) as f32;
                if let Some(d) = data.get_mut(c * plane + o) {
                    *d = (v - 127.5) / 127.5;
                }
            }
            if let Some(d) = data.get_mut(3 * plane + o) {
                *d = trimap.get(i).copied().unwrap_or(0) as f32 / 255.0;
            }
        }
    }
    Ok((data, (hp, wp)))
}

/// A loaded ViTMatte-S model.
pub struct ViTMatte {
    device: Device,
    backbone: VitDet,
    decoder: Decoder,
}

fn matte_err(e: Error) -> Error {
    match e {
        Error::Model(m) => Error::Matte(m),
        other => other,
    }
}

impl ViTMatte {
    /// Load from a directory holding `model.safetensors` (the `hustvl/vitmatte-small-composition-1k`
    /// checkpoint in Hugging Face format).
    pub fn load_on(dir: &Path, device: &Device) -> Result<Self> {
        if !is_matte_model_dir(dir) {
            return Err(Error::Matte(format!("ViTMatte-S weights not found in {} (expected {MATTE_WEIGHTS_FILE})", dir.display())));
        }
        let w: Arc<weights::Weights> = weights::Weights::open(&dir.join(MATTE_WEIGHTS_FILE)).map_err(matte_err)?;
        let vb = VarBuilder::from_backend(Box::new(weights::Backend(w)), DType::F32, device.clone());
        let backbone = VitDet::new(vb.pp("backbone")).map_err(matte_err)?;
        let decoder = Decoder::new(vb.pp("decoder")).map_err(matte_err)?;
        Ok(Self { device: device.clone(), backbone, decoder })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The model input for an 8-bit RGB image and trimap, see [`prepare_input`]: `[1, 4, H, W]`. Also returns `(H, W)`.
    #[doc(hidden)]
    pub fn preprocess(&self, rgb: &[u8], trimap: &[u8], w: usize, h: usize) -> Result<(Tensor, (usize, usize))> {
        let (data, (hp, wp)) = prepare_input(rgb, trimap, w, h)?;
        Ok((Tensor::from_vec(data, (1, IN_CHANNELS, hp, wp), &self.device)?, (hp, wp)))
    }

    /// Every intermediate result (`hidden_0 … hidden_12`, `detail_1 … 3`, `fusion_0 … 3`, `alphas`)
    /// for a preprocessed input, for the reference test.
    #[doc(hidden)]
    pub fn forward_stages(&self, pixels: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let mut stages = Vec::new();
        let features = self.backbone.forward(pixels, Some(&mut stages))?;
        self.decoder.forward(&features, pixels, Some(&mut stages))?;
        Ok(stages)
    }

    /// Alpha (0..1, row-major `w × h`) of the image. `rgb` is 8-bit RGB, `trimap` 8-bit with 0 for
    /// sure background, 255 for sure foreground and anything between for unknown (the model was
    /// trained with 128). The input is padded to multiples of 32 and cropped back; the padded sides
    /// may not exceed [`MATTE_MAX_SIDE`].
    pub fn matte(&self, rgb: &[u8], trimap: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        let (pixels, _) = self.preprocess(rgb, trimap, w, h)?;
        let features = self.backbone.forward(&pixels, None)?;
        let alpha = self.decoder.forward(&features, &pixels, None)?;
        let alpha = alpha.narrow(2, 0, h)?.narrow(3, 0, w)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(alpha.into_iter().map(|a| if a.is_finite() { a.clamp(0.0, 1.0) } else { 0.0 }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("lc-vitmatte-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_missing_model_is_an_error() {
        let d = tmp("missing");
        assert!(!is_matte_model_dir(&d));
        let e = ViTMatte::load_on(&d, &Device::Cpu).err().unwrap().to_string();
        assert!(e.contains("ViTMatte") && e.contains("not found"), "{e}");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn truncated_weights_are_an_error() {
        let d = tmp("truncated");
        // the header promises 1 MB of tensor data; the file stops after 16 bytes
        let header = r#"{"backbone.embeddings.projection.bias":{"dtype":"F32","shape":[250000],"data_offsets":[0,1000000]}}"#;
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header.as_bytes());
        b.extend([0u8; 16]);
        std::fs::write(d.join(MATTE_WEIGHTS_FILE), b).unwrap();
        assert!(is_matte_model_dir(&d));
        let e = ViTMatte::load_on(&d, &Device::Cpu).err().unwrap().to_string();
        assert!(e.starts_with("ViTMatte model:") && e.contains("outside the file"), "{e}");
        // garbage is not a safetensors file either
        std::fs::write(d.join(MATTE_WEIGHTS_FILE), b"not a model").unwrap();
        assert!(ViTMatte::load_on(&d, &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_checkpoint_without_the_tensors_is_an_error() {
        let d = tmp("empty");
        let header = "{}";
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header.as_bytes());
        std::fs::write(d.join(MATTE_WEIGHTS_FILE), b).unwrap();
        assert!(ViTMatte::load_on(&d, &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn bad_dimensions_are_errors_not_panics() {
        assert!(prepare_input(&[], &[], 0, 0).is_err());
        assert!(prepare_input(&[0; 12], &[0; 4], 0, 4).is_err());
        assert!(prepare_input(&[0; 11], &[0; 4], 2, 2).is_err());
        assert!(prepare_input(&[0; 12], &[0; 3], 2, 2).is_err());
        // overflow of w * h
        assert!(prepare_input(&[0; 12], &[0; 4], usize::MAX, 2).is_err());
        assert!(prepare_input(&[], &[], usize::MAX, usize::MAX).is_err());
        // beyond the size limit: the check is before any allocation sized by the input
        let e = prepare_input(&[], &[], 100_000, 100_000).err().unwrap().to_string();
        assert!(e.contains("needs"), "{e}");
        let big = vec![0u8; 1100 * 4];
        assert!(prepare_input(&big[..3300], &big[..1100], 1100, 1).unwrap_err().to_string().contains("limit"));
    }

    #[test]
    fn input_is_normalised_and_padded_to_32() {
        let (data, (hp, wp)) = prepare_input(&[255, 0, 128, 0, 0, 0], &[0, 128], 2, 1).unwrap();
        assert_eq!((hp, wp), (32, 32));
        let plane = 32 * 32;
        assert_eq!(data[0], 1.0);
        assert_eq!(data[plane], -1.0);
        assert!((data[2 * plane] - 1.0 / 255.0).abs() < 1e-6);
        assert!((data[3 * plane + 1] - 128.0 / 255.0).abs() < 1e-6);
        // padding is zero in every channel
        assert!((0..4).all(|c| data[c * plane + 2] == 0.0 && data[c * plane + 32] == 0.0));
    }
}
