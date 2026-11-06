//! The ViTDet backbone of ViTMatte-S: a plain 12-layer ViT (hidden 384, 6 heads, patch 16) over an
//! RGB + trimap input, with windowed attention (14 × 14) in most layers, global attention with
//! decomposed relative position embeddings in the rest, and a residual bottleneck block after
//! each global layer. The absolute position embedding is resized (bicubic) to the input's grid.
//!
//! Modified work (Apache License 2.0, §4(b)): ported by the LightCraft contributors in 2026 from
//! the Python/PyTorch ViTDet code of Hugging Face Transformers (`models/vitdet/modeling_vitdet.py`),
//! Copyright The HuggingFace Team; translated to Rust on candle and restructured. See NOTICE.

use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::nn::{linear, ln, ln_channels};
use crate::{Error, Result};

pub const IN_CHANNELS: usize = 4;
pub const PATCH: usize = 16;
pub const HIDDEN: usize = 384;
const HEADS: usize = 6;
const HEAD_DIM: usize = HIDDEN / HEADS;
pub const LAYERS: usize = 12;
const WINDOW: usize = 14;
/// Layers with global attention (each followed by a residual block).
const GLOBAL: [usize; 4] = [2, 5, 8, 11];
/// Grid of the position embedding as stored: 224 / 16 patches per side.
const PRETRAIN_GRID: usize = 224 / PATCH;
/// Grid the global layers' relative position tables were made for: 512 / 16.
const TRAIN_GRID: usize = 512 / PATCH;
const EPS: f64 = 1e-6;
/// Largest attention-score block computed at once (elements).
const SCORE_BLOCK: usize = 16 << 20;

fn is_global(layer: usize) -> bool {
    GLOBAL.contains(&layer)
}

/// Bicubic resize (PyTorch `mode="bicubic", align_corners=False`, no antialiasing) of an
/// `n × n × c` grid (channels last) to `oh × ow`.
fn resize_bicubic(src: &[f32], n: usize, c: usize, oh: usize, ow: usize) -> Vec<f32> {
    const A: f64 = -0.75;
    let cc1 = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let cc2 = |x: f64| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    // per output index: the four clamped source indices and their weights
    let taps = |out: usize| -> Vec<([usize; 4], [f64; 4])> {
        let scale = n as f64 / out as f64;
        (0..out)
            .map(|i| {
                let s = scale * (i as f64 + 0.5) - 0.5;
                let f = s.floor();
                let t = s - f;
                let w = [cc2(t + 1.0), cc1(t), cc1(1.0 - t), cc2(2.0 - t)];
                let idx = std::array::from_fn(|k| (f as i64 - 1 + k as i64).clamp(0, n as i64 - 1) as usize);
                (idx, w)
            })
            .collect()
    };
    let (ty, tx) = (taps(oh), taps(ow));
    let at = |y: usize, x: usize, ch: usize| src.get((y * n + x) * c + ch).copied().unwrap_or(0.0) as f64;
    let mut out = Vec::with_capacity(oh * ow * c);
    for (yi, wy) in &ty {
        for (xi, wx) in &tx {
            for ch in 0..c {
                let mut v = 0.0;
                for (y, wy) in yi.iter().zip(wy) {
                    let mut row = 0.0;
                    for (x, wx) in xi.iter().zip(wx) {
                        row += at(*y, *x, ch) * wx;
                    }
                    v += row * wy;
                }
                out.push(v as f32);
            }
        }
    }
    out
}

/// Linear resize (PyTorch `mode="linear", align_corners=False`) of `len × c` rows to `out × c`.
fn resize_linear(src: &[f32], len: usize, c: usize, out: usize) -> Vec<f32> {
    let scale = len as f64 / out as f64;
    let mut res = Vec::with_capacity(out * c);
    for i in 0..out {
        let s = (scale * (i as f64 + 0.5) - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(len.saturating_sub(1));
        let i1 = (i0 + 1).min(len.saturating_sub(1));
        let l = s - i0 as f64;
        for ch in 0..c {
            let a = src.get(i0 * c + ch).copied().unwrap_or(0.0) as f64;
            let b = src.get(i1 * c + ch).copied().unwrap_or(0.0) as f64;
            res.push((a * (1.0 - l) + b * l) as f32);
        }
    }
    res
}

/// `get_rel_pos` for equal query and key sizes: the `[size, size, head_dim]` table whose entry
/// `(q, k)` is the embedding of the relative distance `q − k`.
fn rel_table(rel: &[f32], len: usize, size: usize, device: &Device) -> Result<Tensor> {
    let max_rel = 2 * size - 1;
    let resized;
    let rel = if len != max_rel {
        resized = resize_linear(rel, len, HEAD_DIM, max_rel);
        &resized[..]
    } else {
        rel
    };
    let mut data = Vec::with_capacity(size * size * HEAD_DIM);
    for q in 0..size {
        for k in 0..size {
            let r = q + size - 1 - k;
            data.extend((0..HEAD_DIM).map(|c| rel.get(r * HEAD_DIM + c).copied().unwrap_or(0.0)));
        }
    }
    Ok(Tensor::from_vec(data, (size, size, HEAD_DIM), device)?)
}

/// Multi-head attention with decomposed relative position embeddings (`VitDetAttention`).
struct Attention {
    qkv: Linear,
    proj: Linear,
    rel_h: Vec<f32>,
    rel_w: Vec<f32>,
    rel_len: usize,
    /// For windowed layers the tables never change.
    window_tables: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(window: bool, vb: VarBuilder) -> Result<Self> {
        let grid = if window { WINDOW } else { TRAIN_GRID };
        let rel_len = 2 * grid - 1;
        let rel_h = vb.get((rel_len, HEAD_DIM), "rel_pos_h")?.flatten_all()?.to_vec1::<f32>()?;
        let rel_w = vb.get((rel_len, HEAD_DIM), "rel_pos_w")?.flatten_all()?.to_vec1::<f32>()?;
        let window_tables =
            if window { Some((rel_table(&rel_h, rel_len, WINDOW, vb.device())?, rel_table(&rel_w, rel_len, WINDOW, vb.device())?)) } else { None };
        Ok(Self {
            qkv: linear(HIDDEN, 3 * HIDDEN, vb.pp("qkv"))?,
            proj: linear(HIDDEN, HIDDEN, vb.pp("proj"))?,
            rel_h,
            rel_w,
            rel_len,
            window_tables,
        })
    }

    /// `x`: `[B, H, W, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = x.dims4()?;
        let l = h * w;
        let qkv = self.qkv.forward(x)?.reshape((b, l, 3, HEADS, HEAD_DIM))?.permute((2, 0, 3, 1, 4))?.contiguous()?;
        let qkv = qkv.reshape((3, b * HEADS, l, HEAD_DIM))?;
        let (q, k, v) = (qkv.i(0)?, qkv.i(1)?, qkv.i(2)?);
        let (th, tw) = match &self.window_tables {
            Some((a, b)) => (a.clone(), b.clone()),
            None => (rel_table(&self.rel_h, self.rel_len, h, x.device())?, rel_table(&self.rel_w, self.rel_len, w, x.device())?),
        };
        let bn = b * HEADS;
        let chunk = (SCORE_BLOCK / (l * l).max(1)).clamp(1, bn);
        let scale = (HEAD_DIM as f64).powf(-0.5);
        let mut outs = Vec::with_capacity(bn.div_ceil(chunk));
        let mut start = 0;
        while start < bn {
            let n = chunk.min(bn - start);
            let (qc, kc, vc) = (q.narrow(0, start, n)?, k.narrow(0, start, n)?, v.narrow(0, start, n)?);
            let att = (&qc * scale)?.matmul(&kc.t()?)?;
            let att = add_rel(&att, &qc, &th, &tw, h, w)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            outs.push(att.matmul(&vc)?);
            start += n;
        }
        let o = Tensor::cat(&outs, 0)?;
        let o = o.reshape((b, HEADS, h, w, HEAD_DIM))?.permute((0, 2, 3, 1, 4))?.contiguous()?.reshape((b, h, w, HIDDEN))?;
        Ok(self.proj.forward(&o)?)
    }
}

/// `add_decomposed_relative_positions`: `att` is `[n, L, L]`, `q` the unscaled queries `[n, L, d]`.
fn add_rel(att: &Tensor, q: &Tensor, th: &Tensor, tw: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let n = q.dim(0)?;
    let r = q.reshape((n, h, w, HEAD_DIM))?;
    // along height: rh[b, y, x, k] = Σ_c r[b, y, x, c] · th[y, k, c]
    let rh = r.permute((1, 0, 2, 3))?.contiguous()?.reshape((h, n * w, HEAD_DIM))?.matmul(&th.transpose(1, 2)?.contiguous()?)?;
    let rh = rh.reshape((h, n, w, h))?.permute((1, 0, 2, 3))?;
    // along width: rw[b, y, x, k] = Σ_c r[b, y, x, c] · tw[x, k, c]
    let rw = r.permute((2, 0, 1, 3))?.contiguous()?.reshape((w, n * h, HEAD_DIM))?.matmul(&tw.transpose(1, 2)?.contiguous()?)?;
    let rw = rw.reshape((w, n, h, w))?.permute((1, 2, 0, 3))?;
    let a = att.reshape((n, h, w, h, w))?;
    let a = a.broadcast_add(&rh.unsqueeze(4)?)?.broadcast_add(&rw.unsqueeze(3)?)?;
    Ok(a.reshape((n, h * w, h * w))?)
}

/// `[B, H, W, C]` → windows `[B·nh·nw, ws, ws, C]` (zero padded at the bottom and right) and the padded size.
fn window_partition(x: &Tensor, ws: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let (ph, pw) = ((ws - h % ws) % ws, (ws - w % ws) % ws);
    let x = if ph > 0 { x.pad_with_zeros(1, 0, ph)? } else { x.clone() };
    let x = if pw > 0 { x.pad_with_zeros(2, 0, pw)? } else { x };
    let (hp, wp) = (h + ph, w + pw);
    let x = x.reshape((b, hp / ws, ws, wp / ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
    Ok((x.reshape((b * (hp / ws) * (wp / ws), ws, ws, c))?, (hp, wp)))
}

fn window_unpartition(win: &Tensor, ws: usize, (hp, wp): (usize, usize), (h, w): (usize, usize)) -> Result<Tensor> {
    let (nh, nw) = (hp / ws, wp / ws);
    let b = win.dim(0)? / (nh * nw).max(1);
    let c = win.dim(3)?;
    let x = win.reshape((b, nh, nw, ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, hp, wp, c))?;
    Ok(x.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()?)
}

/// `VitDetResBottleneckBlock`: 1×1 → 3×3 → 1×1 convolutions, each followed by a channel layer norm.
struct Residual {
    conv1: Conv2d,
    norm1: LayerNorm,
    conv2: Conv2d,
    norm2: LayerNorm,
    conv3: Conv2d,
    norm3: LayerNorm,
}

impl Residual {
    fn new(vb: VarBuilder) -> Result<Self> {
        let mid = HIDDEN / 2;
        let k3 = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv1: candle_nn::conv2d_no_bias(HIDDEN, mid, 1, Default::default(), vb.pp("conv1"))?,
            norm1: ln(mid, EPS, vb.pp("norm1"))?,
            conv2: candle_nn::conv2d_no_bias(mid, mid, 3, k3, vb.pp("conv2"))?,
            norm2: ln(mid, EPS, vb.pp("norm2"))?,
            conv3: candle_nn::conv2d_no_bias(mid, HIDDEN, 1, Default::default(), vb.pp("conv3"))?,
            norm3: ln(HIDDEN, EPS, vb.pp("norm3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let o = ln_channels(&self.conv1.forward(x)?, &self.norm1)?.gelu_erf()?;
        let o = ln_channels(&self.conv2.forward(&o)?, &self.norm2)?.gelu_erf()?;
        let o = ln_channels(&self.conv3.forward(&o)?, &self.norm3)?;
        Ok((x + o)?)
    }
}

struct Layer {
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    window: bool,
    residual: Option<Residual>,
}

impl Layer {
    fn new(i: usize, vb: VarBuilder) -> Result<Self> {
        let window = !is_global(i);
        Ok(Self {
            norm1: ln(HIDDEN, EPS, vb.pp("norm1"))?,
            attention: Attention::new(window, vb.pp("attention"))?,
            norm2: ln(HIDDEN, EPS, vb.pp("norm2"))?,
            fc1: linear(HIDDEN, 4 * HIDDEN, vb.pp("mlp.fc1"))?,
            fc2: linear(4 * HIDDEN, HIDDEN, vb.pp("mlp.fc2"))?,
            window,
            residual: if window { None } else { Some(Residual::new(vb.pp("residual"))?) },
        })
    }

    /// `x`: `[B, H, W, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w, _) = x.dims4()?;
        let y = self.norm1.forward(x)?;
        let y = if self.window {
            let (win, padded) = window_partition(&y, WINDOW)?;
            window_unpartition(&self.attention.forward(&win)?, WINDOW, padded, (h, w))?
        } else {
            self.attention.forward(&y)?
        };
        let x = (x + y)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        let x = (&x + m)?;
        match &self.residual {
            Some(r) => Ok(r.forward(&x.permute((0, 3, 1, 2))?.contiguous()?)?.permute((0, 2, 3, 1))?.contiguous()?),
            None => Ok(x),
        }
    }
}

pub struct VitDet {
    projection: Conv2d,
    /// `[197, 384]` as stored (the first row is the class token's, unused).
    pos: Vec<f32>,
    layers: Vec<Layer>,
}

impl VitDet {
    /// `vb` at the backbone's prefix (`backbone`).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let e = vb.pp("embeddings");
        let cfg = Conv2dConfig { stride: PATCH, ..Default::default() };
        let projection = candle_nn::conv2d(IN_CHANNELS, HIDDEN, PATCH, cfg, e.pp("projection"))?;
        let pos = e.get((1, PRETRAIN_GRID * PRETRAIN_GRID + 1, HIDDEN), "position_embeddings")?.flatten_all()?.to_vec1::<f32>()?;
        let enc = vb.pp("encoder.layer");
        let layers = (0..LAYERS).map(|i| Layer::new(i, enc.pp(i.to_string()))).collect::<Result<Vec<_>>>()?;
        Ok(Self { projection, pos, layers })
    }

    /// Patch embedding plus the absolute position embedding resized to the grid: `[1, 384, h, w]`.
    fn embed(&self, pixels: &Tensor) -> Result<Tensor> {
        let x = self.projection.forward(pixels)?;
        let (_, _, h, w) = x.dims4()?;
        let grid = self.pos.get(HIDDEN..).ok_or_else(|| Error::Matte("position embedding is too short".into()))?;
        let pos = if h == PRETRAIN_GRID && w == PRETRAIN_GRID { grid.to_vec() } else { resize_bicubic(grid, PRETRAIN_GRID, HIDDEN, h, w) };
        let pos = Tensor::from_vec(pos, (1, h, w, HIDDEN), x.device())?;
        Ok(x.permute((0, 2, 3, 1))?.broadcast_add(&pos)?)
    }

    /// The stride-16 feature map `[1, 384, h/16, w/16]` of `pixels` (`[1, 4, H, W]`, H and W
    /// multiples of 16). With `stages`, the embedding output and every layer's output (NCHW) are
    /// appended as `hidden_0 … hidden_12`.
    pub fn forward(&self, pixels: &Tensor, mut stages: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor> {
        let nchw = |t: &Tensor| -> Result<Tensor> { Ok(t.permute((0, 3, 1, 2))?.contiguous()?) };
        let mut x = self.embed(pixels)?;
        if let Some(s) = stages.as_deref_mut() {
            s.push(("hidden_0".into(), nchw(&x)?));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if let Some(s) = stages.as_deref_mut() {
                s.push((format!("hidden_{}", i + 1), nchw(&x)?));
            }
        }
        nchw(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_of_the_same_size_is_the_identity() {
        let src: Vec<f32> = (0..14 * 14 * 2).map(|i| (i as f32 * 0.37).sin()).collect();
        let out = resize_bicubic(&src, 14, 2, 14, 14);
        assert!(out.iter().zip(&src).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn bicubic_keeps_a_constant_constant() {
        let out = resize_bicubic(&vec![0.5; 14 * 14], 14, 1, 9, 23);
        assert_eq!(out.len(), 9 * 23);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn linear_resize_interpolates() {
        // 3 rows → 5: the middle one stays, the ends are clamped
        let out = resize_linear(&[0.0, 1.0, 2.0], 3, 1, 5);
        assert_eq!(out.len(), 5);
        assert!((out[2] - 1.0).abs() < 1e-6 && out[0] == 0.0 && out[4] == 2.0);
    }

    #[test]
    fn rel_table_indexes_by_distance() {
        let rel: Vec<f32> = (0..7).flat_map(|r| std::iter::repeat_n(r as f32, HEAD_DIM)).collect();
        let t = rel_table(&rel, 7, 4, &Device::Cpu).unwrap();
        let v = t.to_vec3::<f32>().unwrap();
        // distance q − k = 0 is row size − 1 = 3
        assert_eq!(v[2][2][0], 3.0);
        assert_eq!(v[3][0][0], 6.0);
        assert_eq!(v[0][3][0], 0.0);
    }

    #[test]
    fn windows_round_trip_with_padding() {
        let x = Tensor::arange(0f32, (2 * 20 * 17 * 3) as f32, &Device::Cpu).unwrap().reshape((2, 20, 17, 3)).unwrap();
        let (win, padded) = window_partition(&x, 14).unwrap();
        assert_eq!(padded, (28, 28));
        assert_eq!(win.dims(), &[2 * 4, 14, 14, 3]);
        let back = window_unpartition(&win, 14, padded, (20, 17)).unwrap();
        let (a, b) = (x.flatten_all().unwrap().to_vec1::<f32>().unwrap(), back.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        assert_eq!(a, b);
    }
}
