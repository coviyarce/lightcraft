//! The ViTMatte-S port against the reference implementation (Hugging Face `transformers`, fp32),
//! stage by stage: the preprocessed input, the embedding and every transformer layer, the detail
//! stream, the four fusion blocks and the alpha.
//!
//! Needs the checkpoint and a reference file made by `tools/vitmatte_reference.py`:
//! `LIGHTCRAFT_VITMATTE_DIR=<model dir> LIGHTCRAFT_VITMATTE_REF=<ref.safetensors> cargo test -p lightcraft-segment --release -- --nocapture`.
//! Skipped (passes) when either is missing.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lightcraft_segment::ViTMatte;

fn setup() -> Option<(PathBuf, HashMap<String, Tensor>)> {
    let dir = PathBuf::from(std::env::var_os("LIGHTCRAFT_VITMATTE_DIR")?);
    let reference = PathBuf::from(std::env::var_os("LIGHTCRAFT_VITMATTE_REF")?);
    let r = candle_core::safetensors::load(&reference, &Device::Cpu).ok()?;
    Some((dir, r))
}

fn vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Max |a − b| relative to the reference's max |b|.
fn rel_err(name: &str, a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "{name}: shape");
    let (a, b) = (vec(a), vec(b));
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let err = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("{name:>14}: max abs err {err:.2e}, rel {:.2e} (scale {scale:.2e})", err / scale);
    err / scale
}

#[test]
fn matches_reference() {
    let Some((dir, r)) = setup() else {
        lightcraft_testkit::skip!(Model, "set LIGHTCRAFT_VITMATTE_DIR and LIGHTCRAFT_VITMATTE_REF");
        return;
    };
    let model = ViTMatte::load_on(&dir, &Device::Cpu).unwrap();
    for (case, w, h) in [("a", 201usize, 150usize), ("b", 512, 512)] {
        eprintln!("-- case {case}: {w}x{h}");
        let bytes = |k: &str| r[&format!("{case}_{k}")].flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let (rgb, trimap) = (bytes("rgb"), bytes("trimap"));
        let (pixels, _) = model.preprocess(&rgb, &trimap, w, h).unwrap();
        let mut worst = ("", 0f32);
        assert!(rel_err("pixel_values", &pixels, &r[&format!("{case}_pixel_values")]) < 1e-5);
        let t = std::time::Instant::now();
        let stages = model.forward_stages(&pixels).unwrap();
        eprintln!("forward with every stage kept: {:?}", t.elapsed());
        assert_eq!(stages.len(), 13 + 3 + 4 + 1, "stage count");
        let mut worst_name = String::new();
        for (name, ours) in &stages {
            let e = rel_err(name, ours, &r[&format!("{case}_{name}")]);
            if e > worst.1 {
                worst = ("", e);
                worst_name = name.clone();
            }
            assert!(e < 1e-4, "{case} {name}: relative error {e:.2e}");
        }
        eprintln!("worst stage: {worst_name} at {:.2e}", worst.1);
        // the public entry point: padding cropped away, same alpha as the reference
        let alpha = model.matte(&rgb, &trimap, w, h).unwrap();
        assert_eq!(alpha.len(), w * h);
        let want = vec(&r[&format!("{case}_alphas")]);
        let wp = want.len() / (h.div_ceil(32) * 32);
        let err = (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).fold(0f32, |m, (y, x)| m.max((alpha[y * w + x] - want[y * wp + x]).abs()));
        eprintln!("matte(): max abs alpha error {err:.2e}");
        assert!(err < 1e-4);
        let t = std::time::Instant::now();
        model.matte(&rgb, &trimap, w, h).unwrap();
        eprintln!("matte() {w}x{h}: {:?}", t.elapsed());
    }
}
