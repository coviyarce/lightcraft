//! Runs ViTMatte-S on this machine and times it. With an image and a trimap (binary PPM `P6`
//! and PGM `P5`, 8-bit, same size) it writes the alpha as a PGM; without them it times a synthetic
//! 512 × 512 tile (a disc with an unknown ring):
//!
//! `cargo run -p lightcraft-segment --release --example vitmatte_run -- <model dir> [image.ppm trimap.pgm alpha.pgm]`

use std::path::Path;

/// Reads a binary PNM (`P5` or `P6`, maxval 255): its samples and size.
fn read_pnm(path: &Path, magic: &[u8; 2]) -> Result<(Vec<u8>, usize, usize), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.get(..2) != Some(&magic[..]) {
        return Err(format!("{}: not a binary {}", path.display(), String::from_utf8_lossy(magic)).into());
    }
    let mut fields = Vec::new();
    let mut pos = 2;
    while fields.len() < 3 {
        while bytes.get(pos).is_some_and(|b| b.is_ascii_whitespace()) {
            pos += 1;
        }
        if bytes.get(pos) == Some(&b'#') {
            while bytes.get(pos).is_some_and(|b| *b != b'\n') {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while bytes.get(pos).is_some_and(|b| b.is_ascii_digit()) {
            pos += 1;
        }
        fields.push(std::str::from_utf8(bytes.get(start..pos).ok_or("bad header")?)?.parse::<usize>()?);
    }
    let [w, h, max] = fields[..] else { return Err("bad header".into()) };
    if max != 255 {
        return Err("only 8-bit PNM is supported".into());
    }
    let channels = if magic == b"P6" { 3 } else { 1 };
    let data = bytes.get(pos + 1..pos + 1 + w * h * channels).ok_or("truncated file")?;
    Ok((data.to_vec(), w, h))
}

fn synthetic(side: usize) -> (Vec<u8>, Vec<u8>) {
    let (mut rgb, mut tri) = (Vec::new(), Vec::new());
    let c = side as f32 / 2.0;
    for y in 0..side {
        for x in 0..side {
            let r = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt() / (side as f32 / 4.0);
            let fg = r < 1.0;
            rgb.extend(if fg { [200u8, 150, 120] } else { [40, (x * 255 / side) as u8, 90] });
            tri.push(if (0.8..1.2).contains(&r) {
                128
            } else if fg {
                255
            } else {
                0
            });
        }
    }
    (rgb, tri)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().ok_or("usage: vitmatte_run <model dir> [image.ppm trimap.pgm alpha.pgm]")?);
    let t = std::time::Instant::now();
    let model = lightcraft_segment::ViTMatte::load_on(&dir, &candle_core::Device::Cpu)?;
    eprintln!("load: {:?}", t.elapsed());
    let (rgb, tri, w, h, out) = match (args.next(), args.next(), args.next()) {
        (Some(i), Some(t), Some(o)) => {
            let (rgb, w, h) = read_pnm(Path::new(&i), b"P6")?;
            let (tri, tw, th) = read_pnm(Path::new(&t), b"P5")?;
            if (tw, th) != (w, h) {
                return Err("image and trimap differ in size".into());
            }
            (rgb, tri, w, h, Some(o))
        }
        _ => {
            let (rgb, tri) = synthetic(512);
            (rgb, tri, 512, 512, None)
        }
    };
    let mut alpha = Vec::new();
    for run in 0..3 {
        let t = std::time::Instant::now();
        alpha = model.matte(&rgb, &tri, w, h)?;
        eprintln!("run {run}: {w}x{h} in {:?}", t.elapsed());
    }
    if let Some(out) = out {
        let mut f = format!("P5\n{w} {h}\n255\n").into_bytes();
        f.extend(alpha.iter().map(|a| (a * 255.0 + 0.5) as u8));
        std::fs::write(out, f)?;
    }
    Ok(())
}
