//! Where an AI mask's edge needs alpha matting, and what to tell the matting model about it.
//!
//! A stored selection (`SegMask`: SAM 3's coarse 288² logits plus up to a few zoomed-in
//! patches) is accurate to about a grid cell, which is 20 px or more on a 26 MP photo. Its edge
//! band, the part where the model was not sure, is where hair and fabric fibres live. This module
//! finds that band, covers it with square full-resolution tiles (the caller renders each one and
//! runs the matting model on it), draws the trimap the model needs for a tile, and blends
//! the finished tiles where they overlap. It has no model and no I/O: everything here is
//! plain arithmetic on the stored logits, so it is tested without the weights.
//!
//! Coordinates: a stored selection and a [`Tile::rect`] are normalized over the uncropped,
//! oriented photo, the same frame the masks use ([`crate::masks`]); [`Tile::origin`] and
//! [`Tile::px`] are the same tile in pixels of a photo `photo_px` wide and high.

use lightcraft_develop::SegMask;
use lightcraft_develop::segmask::{MAX_DETAIL, MAX_MATTE_TILES};
use lightcraft_geom::Point;

use crate::masks::SegGrid;

/// Side of a matting tile in photo pixels (smaller on a photo that is smaller than this).
pub const TILE: usize = 512;
/// How far neighbouring tiles overlap, in pixels: the width of the feathered seam.
pub const OVERLAP: usize = 32;
/// Photos smaller than this on either side are not matted.
const MIN_PHOTO: usize = 64;
/// Largest photo side accepted (hostile metadata caps the allocations below).
const MAX_PHOTO: usize = 1 << 16;
/// Long side of the grid the band is worked out on (about a quarter of the coarse cell on a
/// 26 MP photo).
const WORK_LONG: usize = 1152;
/// A logit below this in magnitude is "not sure" (a probability of 0.12 to 0.88).
const UNKNOWN_LOGIT: f32 = 2.0;
/// A logit beyond this in magnitude is sure foreground (+) or background (-) in the trimap.
const SURE_LOGIT: f32 = 4.0;
/// The band reaches this many coarse cells either side of the uncertain region ...
const BAND_CELLS: f64 = 1.5;
/// ... plus this many photo pixels (hair that the selection left out).
const HAIR_PX: f64 = 24.0;
/// Largest dilation radius on the working grid.
const MAX_WORK_RADIUS: usize = 96;
/// Enclosed patches of sure area up to this many working cells inside the band become band.
const MAX_HOLE: usize = 64;
/// Erosion radius range of the trimap, in tile pixels.
const TRIMAP_RADIUS: (usize, usize) = (2, 48);
/// Largest tile the trimap and blending handle, in pixels per side.
const MAX_TILE_SIDE: usize = 1024;

/// A square piece of the photo to matte.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    /// The tile in normalized photo coordinates `[x0, y0, x1, y1]`.
    pub rect: [f64; 4],
    /// Its top-left corner, in photo pixels.
    pub origin: (usize, usize),
    /// Its width and height, in photo pixels (equal: tiles are square).
    pub px: (usize, usize),
}

/// The stored selection of one mask component, sampled like the renderer does: zoomed-in patches
/// win (the highest where they overlap) inside their rectangles, the coarse grid is used elsewhere.
struct Field {
    coarse: Option<SegGrid>,
    detail: Vec<SegGrid>,
    /// Photo pixels per cell of the coarsest grid present.
    cell_px: f64,
}

impl Field {
    fn new(seg: Option<&SegMask>, src_detail: &[SegMask], photo: (usize, usize)) -> Option<Field> {
        let coarse = seg.and_then(SegGrid::new);
        let detail: Vec<SegGrid> = src_detail.iter().take(MAX_DETAIL).filter_map(SegGrid::new).collect();
        if coarse.is_none() && detail.is_empty() {
            return None;
        }
        let cell = |m: &SegMask| -> Option<f64> {
            let r = m.bounds()?;
            Some(((r[2] - r[0]) * photo.0 as f64).max((r[3] - r[1]) * photo.1 as f64) / f64::from(m.side.max(1)))
        };
        let cell_px = seg.into_iter().chain(src_detail.iter().take(MAX_DETAIL)).filter_map(cell).fold(0.0f64, f64::max).clamp(1.0, 128.0);
        Some(Field { coarse, detail, cell_px })
    }

    fn logit(&self, n: Point) -> f32 {
        let mut best: Option<f32> = None;
        for d in self.detail.iter().filter(|d| d.contains(n)) {
            let l = d.logit(n);
            best = Some(best.map_or(l, |b| b.max(l)));
        }
        let l = match best {
            Some(l) => l,
            None => self.coarse.as_ref().map_or(-16.0, |c| c.logit(n)),
        };
        if l.is_finite() { l } else { -16.0 }
    }
}

/// Where a selection needs matting, and the trimaps for those places.
pub struct Plan {
    field: Field,
    photo: (usize, usize),
    cell_px: f64,
    /// Cells of the grid the band is worked out on ([`Plan::band`]).
    tile: usize,
    overlap: usize,
}

impl Plan {
    /// `None` when the photo is too small or too large, or no selection decodes (damaged data).
    pub fn new(seg: Option<&SegMask>, detail: &[SegMask], photo: (usize, usize)) -> Option<Plan> {
        Plan::with_tile(seg, detail, photo, TILE, OVERLAP)
    }

    /// [`Plan::new`] with another tile size and overlap (tests; the app uses [`TILE`] and [`OVERLAP`]).
    pub fn with_tile(seg: Option<&SegMask>, detail: &[SegMask], photo: (usize, usize), tile: usize, overlap: usize) -> Option<Plan> {
        if photo.0 < MIN_PHOTO || photo.1 < MIN_PHOTO || photo.0 > MAX_PHOTO || photo.1 > MAX_PHOTO {
            return None;
        }
        let field = Field::new(seg, detail, photo)?;
        let cell_px = field.cell_px;
        let tile = tile.clamp(MIN_PHOTO, MAX_TILE_SIDE).min(photo.0).min(photo.1);
        Some(Plan { field, photo, cell_px, tile, overlap: overlap.min(tile / 2) })
    }

    /// The tiles that cover the edge band: only where the selection is uncertain, at most
    /// [`MAX_MATTE_TILES`] (the ones with the most band first, when there are more), in reading order.
    pub fn tiles(&self) -> Vec<Tile> {
        let Some(band) = self.band() else { return Vec::new() };
        let (pw, ph) = self.photo;
        let ts = self.tile;
        let stride = (ts - self.overlap).max(1);
        let positions = |len: usize| -> Vec<usize> {
            let mut out = Vec::new();
            let mut p = 0usize;
            loop {
                out.push(p.min(len - ts));
                if p + ts >= len {
                    return out;
                }
                p += stride;
            }
        };
        let (xs, ys) = (positions(pw), positions(ph));
        let mut found: Vec<(usize, usize, usize)> = Vec::new();
        for &oy in &ys {
            for &ox in &xs {
                let n = band.count(ox, oy, ts);
                if n > 0 {
                    found.push((n, ox, oy));
                }
            }
        }
        // the most band first; ties in reading order, so the choice is the same every time
        found.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.cmp(&b.2)).then(a.1.cmp(&b.1)));
        found.truncate(MAX_MATTE_TILES);
        found.sort_by_key(|&(_, ox, oy)| (oy, ox));
        found
            .into_iter()
            .map(|(_, ox, oy)| Tile {
                rect: [ox as f64 / pw as f64, oy as f64 / ph as f64, (ox + ts) as f64 / pw as f64, (oy + ts) as f64 / ph as f64],
                origin: (ox, oy),
                px: (ts, ts),
            })
            .collect()
    }

    /// The edge band on the working grid: uncertain cells, grown by [`BAND_CELLS`] coarse cells and
    /// [`HAIR_PX`], closed, with the small enclosed patches filled in.
    fn band(&self) -> Option<Band> {
        let (pw, ph) = self.photo;
        let long = pw.max(ph);
        let s = (long as f64 / WORK_LONG as f64).max(1.0);
        let (gw, gh) = (((pw as f64 / s).ceil() as usize).max(1), ((ph as f64 / s).ceil() as usize).max(1));
        let mut unknown = vec![false; gw * gh];
        for gy in 0..gh {
            for gx in 0..gw {
                let n = Point::new(((gx as f64 + 0.5) * s / pw as f64).min(1.0), ((gy as f64 + 0.5) * s / ph as f64).min(1.0));
                if let Some(c) = unknown.get_mut(gy * gw + gx) {
                    *c = self.field.logit(n).abs() < UNKNOWN_LOGIT;
                }
            }
        }
        let r = (((BAND_CELLS * self.cell_px + HAIR_PX) / s).ceil() as usize).min(MAX_WORK_RADIUS);
        let mut cells = dilate(&unknown, gw, gh, r);
        // close (gaps between strands), then fill small enclosed patches
        let closed = erode(&dilate(&cells, gw, gh, 2), gw, gh, 2);
        for (c, k) in cells.iter_mut().zip(closed) {
            *c = *c || k;
        }
        fill_holes(&mut cells, gw, gh, MAX_HOLE);
        Some(Band::new(&cells, gw, gh, s))
    }

    /// The trimap of `tile` at `w × h` pixels (the size it was actually rendered at): 255 where the
    /// selection is sure foreground, 0 where it is sure background, 128 elsewhere. Empty when the
    /// size is absurd.
    pub fn trimap(&self, tile: &Tile, w: usize, h: usize) -> Vec<u8> {
        if w == 0 || h == 0 || w > MAX_TILE_SIDE || h > MAX_TILE_SIDE || !tile.rect.iter().all(|v| v.is_finite()) {
            return Vec::new();
        }
        let r = tile.rect;
        let mut fg = vec![false; w * h];
        let mut bg = vec![false; w * h];
        for y in 0..h {
            let ny = r[1] + (y as f64 + 0.5) / h as f64 * (r[3] - r[1]);
            for x in 0..w {
                let nx = r[0] + (x as f64 + 0.5) / w as f64 * (r[2] - r[0]);
                let l = self.field.logit(Point::new(nx.clamp(0.0, 1.0), ny.clamp(0.0, 1.0)));
                if let (Some(f), Some(b)) = (fg.get_mut(y * w + x), bg.get_mut(y * w + x)) {
                    *f = l > SURE_LOGIT;
                    *b = l < -SURE_LOGIT;
                }
            }
        }
        let scale = w as f64 / tile.px.0.max(1) as f64;
        let radius = ((BAND_CELLS * self.cell_px * scale).round() as usize).clamp(TRIMAP_RADIUS.0, TRIMAP_RADIUS.1);
        let fg = erode_keeping(&fg, w, h, radius);
        let bg = erode_keeping(&bg, w, h, radius);
        fg.iter()
            .zip(&bg)
            .map(|(f, b)| {
                if *f {
                    255
                } else if *b {
                    0
                } else {
                    128
                }
            })
            .collect()
    }
}

/// The tiles of the edge band of a selection (see [`Plan::tiles`]) on a photo `photo_px` wide and high.
pub fn band_tiles(seg: Option<&SegMask>, detail: &[SegMask], photo_px: (usize, usize)) -> Vec<Tile> {
    Plan::new(seg, detail, photo_px).map(|p| p.tiles()).unwrap_or_default()
}

/// The trimap of `tile` (see [`Plan::trimap`]).
pub fn trimap(seg: Option<&SegMask>, detail: &[SegMask], photo_px: (usize, usize), tile: &Tile, w: usize, h: usize) -> Vec<u8> {
    Plan::new(seg, detail, photo_px).map(|p| p.trimap(tile, w, h)).unwrap_or_default()
}

/// The band as a summed-area table over the working grid, to count a tile's band cells at once.
struct Band {
    sat: Vec<u32>,
    w: usize,
    h: usize,
    /// Photo pixels per working cell.
    s: f64,
}

impl Band {
    fn new(cells: &[bool], w: usize, h: usize, s: f64) -> Band {
        let mut sat = vec![0u32; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0u32;
            for x in 0..w {
                row += u32::from(cells.get(y * w + x).copied().unwrap_or(false));
                let above = sat.get(y * (w + 1) + x + 1).copied().unwrap_or(0);
                if let Some(c) = sat.get_mut((y + 1) * (w + 1) + x + 1) {
                    *c = above + row;
                }
            }
        }
        Band { sat, w, h, s }
    }

    /// Band cells inside the `ts`-pixel square at photo pixel (`ox`, `oy`).
    fn count(&self, ox: usize, oy: usize, ts: usize) -> usize {
        let cell = |p: f64, max: usize| ((p / self.s).floor().max(0.0) as usize).min(max);
        let (x0, y0) = (cell(ox as f64, self.w), cell(oy as f64, self.h));
        let (x1, y1) = (cell(((ox + ts) as f64 / self.s).ceil() * self.s, self.w), cell(((oy + ts) as f64 / self.s).ceil() * self.s, self.h));
        let at = |x: usize, y: usize| self.sat.get(y * (self.w + 1) + x).copied().unwrap_or(0);
        (at(x1, y1) + at(x0, y0)).saturating_sub(at(x1, y0) + at(x0, y1)) as usize
    }
}

/// `m` grown by `r` cells in every direction (a square window), outside the grid counting as empty.
fn dilate(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 || w == 0 || h == 0 || m.len() != w * h {
        return m.to_vec();
    }
    let mut pre = vec![0u32; w.max(h) + 1];
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            pre[x + 1] = pre[x] + u32::from(m[y * w + x]);
        }
        for x in 0..w {
            tmp[y * w + x] = pre[(x + r + 1).min(w)] > pre[x.saturating_sub(r)];
        }
    }
    let mut out = vec![false; w * h];
    for x in 0..w {
        for y in 0..h {
            pre[y + 1] = pre[y] + u32::from(tmp[y * w + x]);
        }
        for y in 0..h {
            out[y * w + x] = pre[(y + r + 1).min(h)] > pre[y.saturating_sub(r)];
        }
    }
    out
}

/// `m` shrunk by `r` cells; the edge of the grid does not eat into it (the object goes on beyond it).
fn erode(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let inv: Vec<bool> = m.iter().map(|v| !v).collect();
    dilate(&inv, w, h, r).into_iter().map(|v| !v).collect()
}

/// [`erode`] by `r`, or by less where that would leave nothing of a mask that has something (a
/// thin strand stays sure foreground instead of turning unknown).
fn erode_keeping(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if !m.iter().any(|v| *v) {
        return m.to_vec();
    }
    let mut r = r;
    loop {
        let e = erode(m, w, h, r);
        if r == 0 || e.iter().any(|v| *v) {
            return e;
        }
        r /= 2;
    }
}

/// Mark as band the enclosed patches of non-band cells that are at most `max_area` cells and do not
/// touch the grid's edge (4-connected).
fn fill_holes(band: &mut [bool], w: usize, h: usize, max_area: usize) {
    let mut seen = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if band.get(start).copied().unwrap_or(true) || seen[start] {
            continue;
        }
        let mut cells: Vec<usize> = Vec::new();
        let (mut area, mut border) = (0usize, false);
        seen[start] = true;
        stack.push(start);
        while let Some(i) = stack.pop() {
            area += 1;
            if cells.len() <= max_area {
                cells.push(i);
            }
            let (x, y) = (i % w, i / w);
            border |= x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            let mut visit = |j: usize| {
                if !band[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        if area <= max_area && !border {
            for i in cells {
                band[i] = true;
            }
        }
    }
}

/// `src` (`sw × sh`) resampled to `dw × dh` by bilinear interpolation (for alpha), or an empty
/// vector when a size is zero or `src` does not match its size.
pub fn resize_alpha(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || src.len() != sw.saturating_mul(sh) || dw.saturating_mul(dh) > MAX_TILE_SIDE * MAX_TILE_SIDE {
        return Vec::new();
    }
    if (sw, sh) == (dw, dh) {
        return src.to_vec();
    }
    let at = |x: usize, y: usize| src.get(y.min(sh - 1) * sw + x.min(sw - 1)).copied().unwrap_or(0.0);
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
            let bot = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
            out.push(top * (1.0 - ty) + bot * ty);
        }
    }
    out
}

/// `src` (`sw × sh`) resampled to `dw × dh` by nearest neighbour (for a trimap, whose values must
/// stay 0, 128 and 255), or empty like [`resize_alpha`].
pub fn resize_trimap(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || src.len() != sw.saturating_mul(sh) || dw.saturating_mul(dh) > MAX_TILE_SIDE * MAX_TILE_SIDE {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        let sy = ((y * 2 + 1) * sh / (dh * 2)).min(sh - 1);
        for x in 0..dw {
            let sx = ((x * 2 + 1) * sw / (dw * 2)).min(sw - 1);
            out.push(src.get(sy * sw + sx).copied().unwrap_or(128));
        }
    }
    out
}

/// Make the tiles agree where they overlap: each pixel covered by several tiles becomes the average
/// of their alphas, weighted by how far it is from each tile's border (a ramp over [`OVERLAP`]
/// pixels), so there is no seam. `alphas[i]` is tile `i`'s `px.0 × px.1` alpha; a tile whose alpha has
/// the wrong length is left as it is and ignored by the others. Pixels no other tile covers keep
/// their value exactly.
pub fn feather_overlaps(tiles: &[Tile], alphas: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let valid = |i: usize| tiles.get(i).zip(alphas.get(i)).filter(|(t, a)| t.px.0.saturating_mul(t.px.1) == a.len() && t.px.0 > 0 && t.px.1 > 0);
    let weight = |t: &Tile, lx: usize, ly: usize| {
        let d = lx.min(ly).min(t.px.0 - 1 - lx).min(t.px.1 - 1 - ly) + 1;
        (d as f32 / OVERLAP as f32).min(1.0)
    };
    let mut out: Vec<Vec<f32>> = alphas.to_vec();
    for i in 0..tiles.len().min(alphas.len()) {
        let Some((ti, ai)) = valid(i) else { continue };
        // the other tiles that touch this one
        let others: Vec<usize> = (0..tiles.len().min(alphas.len()))
            .filter(|&j| j != i)
            .filter(|&j| {
                valid(j).is_some_and(|(tj, _)| {
                    tj.origin.0 < ti.origin.0 + ti.px.0
                        && ti.origin.0 < tj.origin.0 + tj.px.0
                        && tj.origin.1 < ti.origin.1 + ti.px.1
                        && ti.origin.1 < tj.origin.1 + tj.px.1
                })
            })
            .collect();
        if others.is_empty() {
            continue;
        }
        let Some(dst) = out.get_mut(i) else { continue };
        for ly in 0..ti.px.1 {
            let gy = ti.origin.1 + ly;
            let rows: Vec<usize> =
                others.iter().copied().filter(|&j| valid(j).is_some_and(|(tj, _)| gy >= tj.origin.1 && gy < tj.origin.1 + tj.px.1)).collect();
            if rows.is_empty() {
                continue;
            }
            for lx in 0..ti.px.0 {
                let gx = ti.origin.0 + lx;
                let w0 = weight(ti, lx, ly);
                let (mut sum, mut total) = (w0 * ai.get(ly * ti.px.0 + lx).copied().unwrap_or(0.0), w0);
                for &j in &rows {
                    let Some((tj, aj)) = valid(j) else { continue };
                    if gx < tj.origin.0 || gx >= tj.origin.0 + tj.px.0 {
                        continue;
                    }
                    let (jx, jy) = (gx - tj.origin.0, gy - tj.origin.1);
                    let w = weight(tj, jx, jy);
                    sum += w * aj.get(jy * tj.px.0 + jx).copied().unwrap_or(0.0);
                    total += w;
                }
                if total > 0.0
                    && let Some(v) = dst.get_mut(ly * ti.px.0 + lx)
                {
                    *v = sum / total;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_develop::segmask::MAX_MATTE_CELLS;

    /// A 288² selection of the normalized box `[x0, y0, x1, y1]`, with a soft one-cell edge.
    fn box_seg(b: [f64; 4]) -> SegMask {
        let side = 288usize;
        let l: Vec<f32> = (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f64 / side as f64 + 0.5 / side as f64, (i / side) as f64 / side as f64 + 0.5 / side as f64);
                let d = (x - b[0]).min(b[2] - x).min(y - b[1]).min(b[3] - y) * side as f64;
                (d * 4.0).clamp(-8.0, 8.0) as f32
            })
            .collect();
        SegMask::from_logits(side, &l)
    }

    fn inside(t: &Tile, b: [f64; 4], margin: f64) -> bool {
        t.rect[0] > b[0] + margin && t.rect[2] < b[2] - margin && t.rect[1] > b[1] + margin && t.rect[3] < b[3] - margin
    }

    #[test]
    fn a_box_selection_gets_tiles_only_around_its_edge() {
        let b = [0.2, 0.2, 0.8, 0.8];
        let tiles = band_tiles(Some(&box_seg(b)), &[], (4000, 3000));
        assert!(tiles.len() > 8 && tiles.len() <= MAX_MATTE_TILES, "{}", tiles.len());
        // none lies deep inside the box (the band is a ring about 100 px wide), none outside it
        assert!(tiles.iter().all(|t| !inside(t, b, 0.1)), "an interior tile");
        for t in &tiles {
            assert!(t.rect.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "{t:?}");
            assert_eq!(t.px, (TILE, TILE));
            assert!(t.origin.0 + t.px.0 <= 4000 && t.origin.1 + t.px.1 <= 3000);
            assert!(t.rect[2] > b[0] - 0.05 && t.rect[0] < b[2] + 0.05, "outside the box: {t:?}");
        }
        // every corner of the box is covered
        for (cx, cy) in [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)] {
            assert!(tiles.iter().any(|t| (t.rect[0]..=t.rect[2]).contains(&cx) && (t.rect[1]..=t.rect[3]).contains(&cy)), "{cx},{cy}");
        }
    }

    #[test]
    fn a_selection_of_most_of_the_photo_still_gets_at_most_the_cap() {
        let tiles = band_tiles(Some(&box_seg([0.02, 0.02, 0.98, 0.98])), &[], (6240, 4160));
        assert_eq!(tiles.len(), MAX_MATTE_TILES, "the perimeter needs more than the cap");
        assert!(tiles.len() * TILE * TILE <= MAX_MATTE_CELLS);
        let mut origins: Vec<_> = tiles.iter().map(|t| t.origin).collect();
        origins.dedup();
        assert_eq!(origins.len(), tiles.len(), "no tile twice");
        assert!(tiles.windows(2).all(|w| (w[0].origin.1, w[0].origin.0) < (w[1].origin.1, w[1].origin.0)), "reading order");
    }

    #[test]
    fn the_choice_is_the_same_every_time() {
        let seg = box_seg([0.02, 0.02, 0.98, 0.98]);
        assert_eq!(band_tiles(Some(&seg), &[], (6240, 4160)), band_tiles(Some(&seg), &[], (6240, 4160)));
    }

    #[test]
    fn damaged_or_absurd_input_gives_no_tiles() {
        let damaged = SegMask { side: 288, data: "?".into(), rect: None, grid: None };
        assert!(band_tiles(Some(&damaged), &[], (4000, 3000)).is_empty());
        let nan = SegMask { rect: Some([f64::NAN, 0.0, 1.0, 1.0]), ..box_seg([0.2, 0.2, 0.8, 0.8]) };
        assert!(band_tiles(Some(&nan), &[], (4000, 3000)).is_empty());
        let inverted = SegMask { rect: Some([0.9, 0.9, 0.1, 0.1]), ..box_seg([0.2, 0.2, 0.8, 0.8]) };
        assert!(band_tiles(Some(&inverted), &[], (4000, 3000)).is_empty());
        assert!(band_tiles(None, &[], (4000, 3000)).is_empty());
        let ok = box_seg([0.2, 0.2, 0.8, 0.8]);
        for photo in [(0, 0), (10, 10), (4000, 0), (usize::MAX, usize::MAX), (1 << 20, 100)] {
            assert!(band_tiles(Some(&ok), &[], photo).is_empty(), "{photo:?}");
        }
        assert!(
            Plan::new(Some(&ok), &[], (4000, 3000))
                .is_some_and(|p| p.trimap(&Tile { rect: [f64::NAN; 4], origin: (0, 0), px: (8, 8) }, 8, 8).is_empty())
        );
        assert!(trimap(Some(&ok), &[], (4000, 3000), &Tile { rect: [0.0, 0.0, 1.0, 1.0], origin: (0, 0), px: (8, 8) }, 0, 5).is_empty());
    }

    #[test]
    fn nothing_selected_needs_no_matting() {
        let nothing = SegMask::from_logits(288, &vec![-8.0; 288 * 288]);
        assert!(band_tiles(Some(&nothing), &[], (4000, 3000)).is_empty());
        let everything = SegMask::from_logits(288, &vec![8.0; 288 * 288]);
        assert!(band_tiles(Some(&everything), &[], (4000, 3000)).is_empty());
    }

    #[test]
    fn a_small_photo_gets_a_smaller_tile() {
        let tiles = band_tiles(Some(&box_seg([0.2, 0.2, 0.8, 0.8])), &[], (300, 200));
        assert!(!tiles.is_empty());
        assert!(tiles.iter().all(|t| t.px == (200, 200) && t.origin.0 + 200 <= 300));
    }

    #[test]
    fn a_detail_patch_inside_its_rectangle_decides_where_the_band_is() {
        // a coarse box, and a patch that says the box's top-left corner area is all inside
        let seg = box_seg([0.2, 0.2, 0.8, 0.8]);
        let l = vec![8.0f32; 64 * 64];
        let patch = SegMask::from_logits_in(64, &l, [0.0, 0.0, 0.5, 0.5]);
        let with = band_tiles(Some(&seg), &[patch], (4000, 3000));
        let without = band_tiles(Some(&seg), &[], (4000, 3000));
        assert!(with.len() < without.len(), "{} vs {}", with.len(), without.len());
    }

    #[test]
    fn dilate_erode_and_holes() {
        let (w, h) = (20, 12);
        let mut m = vec![false; w * h];
        m[6 * w + 10] = true;
        let d = dilate(&m, w, h, 3);
        assert_eq!(d.iter().filter(|v| **v).count(), 49);
        assert_eq!(erode(&d, w, h, 3).iter().filter(|v| **v).count(), 1);
        // the grid's edge does not erode
        assert!(erode(&vec![true; w * h], w, h, 5).iter().all(|v| *v));
        // a small hole is filled, a big one and one at the border are not
        let mut band = vec![true; w * h];
        for (x, y) in [(5, 5), (6, 5)] {
            band[y * w + x] = false;
        }
        band[0] = false;
        for y in 2..10 {
            for x in 12..19 {
                band[y * w + x] = false;
            }
        }
        fill_holes(&mut band, w, h, 4);
        assert!(band[5 * w + 5] && band[5 * w + 6], "small hole filled");
        assert!(!band[0], "the border one stays");
        assert!(!band[5 * w + 14], "the big one stays");
    }

    #[test]
    fn the_trimap_has_three_values_and_follows_the_selection() {
        let b = [0.2, 0.2, 0.8, 0.8];
        let seg = box_seg(b);
        let plan = Plan::new(Some(&seg), &[], (4000, 3000)).unwrap();
        // a tile on the left edge of the box
        let t = Tile { rect: [1000.0 / 4000.0, 1000.0 / 3000.0, 1512.0 / 4000.0, 1512.0 / 3000.0], origin: (1000, 1000), px: (512, 512) };
        let tri = plan.trimap(&t, 512, 512);
        assert_eq!(tri.len(), 512 * 512);
        assert!(tri.iter().all(|v| [0, 128, 255].contains(v)));
        let row = &tri[256 * 512..257 * 512];
        // the box edge is at x = 800 px, which is column -200: the tile is inside the box here, so
        // use the box's left edge instead: tile at x 600..1112
        let _ = row;
        let t = Tile { rect: [600.0 / 4000.0, 1000.0 / 3000.0, 1112.0 / 4000.0, 1512.0 / 3000.0], origin: (600, 1000), px: (512, 512) };
        let tri = plan.trimap(&t, 512, 512);
        let row = &tri[256 * 512..257 * 512];
        assert_eq!(row[0], 0, "outside is sure background");
        assert_eq!(row[511], 255, "inside is sure foreground");
        let first_fg = row.iter().position(|v| *v == 255).unwrap();
        let last_bg = row.iter().rposition(|v| *v == 0).unwrap();
        assert!(first_fg > last_bg + 20, "an unknown band between: {last_bg}..{first_fg}");
        assert!(row[last_bg + 1..first_fg].iter().all(|v| *v == 128));
        // the box edge (x = 800 px = column 200) lies inside the unknown band
        assert!(last_bg < 200 && first_fg > 200, "{last_bg} {first_fg}");
    }

    #[test]
    fn a_thin_selection_keeps_some_sure_foreground() {
        // a vertical strip one coarse cell wide: eroding by the band radius would remove all of it
        let side = 288usize;
        let l: Vec<f32> = (0..side * side).map(|i| if (i % side) == 144 { 8.0 } else { -8.0 }).collect();
        let seg = SegMask::from_logits(side, &l);
        let plan = Plan::new(Some(&seg), &[], (4000, 3000)).unwrap();
        let t = Tile { rect: [0.4, 0.4, 0.6, 0.6], origin: (1600, 1200), px: (512, 512) };
        let tri = plan.trimap(&t, 512, 512);
        assert!(tri.contains(&255), "the strip survives as sure foreground");
        assert!(tri.contains(&0));
        assert!(tri.iter().all(|v| [0, 128, 255].contains(v)));
    }

    #[test]
    fn the_trimap_is_resampled_with_its_three_values() {
        let src: Vec<u8> = (0..16).map(|i| [0, 128, 255][i % 3]).collect();
        let r = resize_trimap(&src, 4, 4, 7, 5);
        assert_eq!(r.len(), 35);
        assert!(r.iter().all(|v| [0, 128, 255].contains(v)));
        assert!(resize_trimap(&src, 4, 4, 0, 5).is_empty() && resize_trimap(&src, 3, 4, 7, 5).is_empty());
        let a = resize_alpha(&[0.0, 1.0, 0.0, 1.0], 2, 2, 4, 4);
        assert_eq!(a.len(), 16);
        assert!(a.iter().all(|v| (0.0..=1.0).contains(v)));
        assert_eq!(resize_alpha(&[0.25; 4], 2, 2, 2, 2), vec![0.25; 4]);
        assert!(resize_alpha(&[0.25; 3], 2, 2, 2, 2).is_empty());
    }

    fn tile_at(x: usize, y: usize, s: usize) -> Tile {
        Tile { rect: [0.0; 4], origin: (x, y), px: (s, s) }
    }

    #[test]
    fn overlapping_tiles_are_blended_without_a_seam() {
        // two 64 px tiles overlapping by 32 px: the left says 0, the right says 1
        let tiles = [tile_at(0, 0, 64), tile_at(32, 0, 64)];
        let alphas = [vec![0.0f32; 64 * 64], vec![1.0f32; 64 * 64]];
        let out = feather_overlaps(&tiles, &alphas);
        // outside the overlap nothing changes
        assert_eq!(out[0][10], 0.0);
        assert_eq!(out[1][50], 1.0);
        // inside, both tiles hold the same values, rising monotonically from 0 to 1 across the seam
        let row = 20 * 64;
        let left: Vec<f32> = (32..64).map(|x| out[0][row + x]).collect();
        let right: Vec<f32> = (0..32).map(|x| out[1][row + x]).collect();
        assert_eq!(left, right);
        assert!(left.windows(2).all(|w| w[1] >= w[0]), "{left:?}");
        assert!(left[0] < 0.2 && left[31] > 0.8, "{} {}", left[0], left[31]);
        // equal alphas stay equal
        let same = feather_overlaps(&tiles, &[vec![0.3; 64 * 64], vec![0.3; 64 * 64]]);
        assert!(same.iter().flatten().all(|v| (v - 0.3).abs() < 1e-6));
        // a tile of the wrong size is left alone and ignored
        let bad = feather_overlaps(&tiles, &[vec![0.0; 5], vec![1.0; 64 * 64]]);
        assert_eq!(bad[0].len(), 5);
        assert!(bad[1].iter().all(|v| *v == 1.0));
        // fewer alphas than tiles
        assert_eq!(feather_overlaps(&tiles, &[vec![0.5; 64 * 64]]).len(), 1);
    }
}
