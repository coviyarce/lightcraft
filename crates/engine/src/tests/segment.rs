//! AI masks without the SAM 3 model (it is never required): every request is a clear error that
//! comes back at once, nothing waits on the network or the model, a damaged model is an error
//! (not a crash or a stuck "busy"), and the download command needs consent and fails cleanly.
//! No test touches the internet: downloads go to a local server.

use std::time::{Duration, Instant};

use serde_json::json;

use crate::Session;
use crate::segment::{NOT_INSTALLED, Segmenter};

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lc-sam3-engine-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// A folder that looks installed but holds no usable model.
pub(super) fn damaged_model(name: &str) -> std::path::PathBuf {
    let d = tmp(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("model.safetensors"), b"\x10\x00\x00\x00\x00\x00\x00\x00{not json at all}").unwrap();
    std::fs::write(d.join("vocab.json"), b"{}").unwrap();
    std::fs::write(d.join("merges.txt"), b"#version: 0.2\n").unwrap();
    d
}

/// Poll until the worker is idle (or `limit` passes), collecting messages.
fn settle(s: &mut Session, limit: Duration) -> Vec<String> {
    let t = Instant::now();
    let mut messages = Vec::new();
    loop {
        let p = s.segment_poll();
        messages.extend(p.messages);
        if !s.segmenter.busy() && !s.segmenter.detail_busy() && !s.segmenter.matte_busy() {
            messages.extend(s.segment_poll().messages);
            return messages;
        }
        assert!(t.elapsed() < limit, "the worker never finished");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn status_and_model_commands_without_the_model() {
    let mut s = Session::with_demo();
    let st = s.execute("segment.model.status", &json!({})).unwrap();
    assert_eq!(st["available"], Segmenter::AVAILABLE);
    assert_eq!(st["installed"], false);
    assert_eq!(st["download"]["running"], false);
    assert!(st["license"].as_str().unwrap().contains("SAM License"));
    // downloading needs the user's consent, and names the size and the licence
    let e = s.execute("segment.model.download", &json!({})).unwrap_err().to_string();
    assert!(e.contains("acknowledged") && e.contains("SAM License") && e.contains("GB"), "{e}");
    let e = s.execute("segment.model.download", &json!({"acknowledged": "yes"})).unwrap_err().to_string();
    assert!(e.contains("acknowledged"), "{e}");
    assert_eq!(s.execute("segment.model.cancel", &json!({})).unwrap()["cancelled"], false);
    if !Segmenter::AVAILABLE {
        assert!(s.execute("segment.model.download", &json!({"acknowledged": true})).is_err());
        return;
    }
    // no folder, then no download location: clear errors, nothing started
    let e = s.execute("segment.model.download", &json!({"acknowledged": true})).unwrap_err().to_string();
    assert!(e.contains("no folder"), "{e}");
    let dir = tmp("nomirrors");
    s.segmenter.dir = Some(dir.clone());
    s.segmenter.mirrors_file = Some(dir.join("none.txt"));
    if s.segmenter.mirrors().is_empty() {
        let e = s.execute("segment.model.download", &json!({"acknowledged": true})).unwrap_err().to_string();
        assert!(e.contains("LIGHTCRAFT_SAM3_MIRRORS"), "{e}");
    }
    assert!(!s.segmenter.download_status().running);
}

#[cfg(feature = "sam")]
#[test]
fn a_failing_download_ends_with_an_error_and_never_blocks() {
    use std::io::{BufRead, BufReader, Write};
    // a local mirror that has nothing; its first answer waits until the test has made its second
    // request, so the download is still running then however the threads are scheduled (a 404 at
    // once could end it before, and the second request would rightly start a new one)
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/sam3", l.local_addr().unwrap());
    let (release, released) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut first = Some(released);
        for mut c in l.incoming().flatten() {
            let mut r = BufReader::new(c.try_clone().unwrap());
            let mut line = String::new();
            while r.read_line(&mut line).unwrap_or(0) > 2 {
                line.clear();
            }
            if let Some(released) = first.take() {
                let _ = released.recv_timeout(Duration::from_secs(10));
            }
            let _ = c.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 13\r\nConnection: close\r\n\r\n<h1>404</h1>\n");
        }
    });
    let dir = tmp("download404");
    let mirrors = dir.with_extension("mirrors.txt");
    std::fs::write(&mirrors, format!("# test mirror\n{base}\n")).unwrap();
    let mut s = Session::with_demo();
    s.segmenter.dir = Some(dir.clone());
    s.segmenter.mirrors_file = Some(mirrors.clone());
    let t = Instant::now();
    let r = s.execute("segment.model.download", &json!({"acknowledged": true})).unwrap();
    assert_eq!(r["started"], true);
    assert!(t.elapsed() < Duration::from_secs(1), "starting a download returns at once");
    // a second request while it runs starts nothing
    let again = s.execute("segment.model.download", &json!({"acknowledged": true})).unwrap();
    assert!(again["started"] == false || again["downloading"] == false);
    let _ = release.send(());
    let t = Instant::now();
    while s.segmenter.download_status().running {
        assert!(t.elapsed() < Duration::from_secs(30), "the download never ended");
        // AI mask requests meanwhile: an error at once, never a wait
        let q = Instant::now();
        let e = s.execute("mask.add", &json!({"kind": "prompt", "text": "sky"})).unwrap_err().to_string();
        assert!(e.contains(NOT_INSTALLED), "{e}");
        assert!(q.elapsed() < Duration::from_millis(500));
        std::thread::sleep(Duration::from_millis(20));
    }
    let st = s.segmenter.download_status();
    assert!(!st.finished);
    assert!(!st.cancelled, "a failure is not a cancel");
    let e = st.error.unwrap();
    assert!(e.contains("not found"), "{e}");
    assert!(!s.segmenter.installed());
    // nothing was left behind under a real name, and no error page was saved
    for f in ["model.safetensors", "vocab.json", "merges.txt", "vocab.json.part"] {
        assert!(!dir.join(f).exists(), "{f}");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&mirrors);
}

/// The app (background mode) never waits: with the model missing every request fails at once;
/// with a damaged model the requests are queued, fail on the worker, and come back as messages.
#[test]
fn background_requests_never_wait_and_failures_come_back_as_messages() {
    let mut s = Session::with_demo();
    s.segmenter.background = true;
    let id = s.active().unwrap();
    s.segmenter.dir = Some(tmp("missing"));
    for (cmd, p) in [("mask.add", json!({"kind": "object"})), ("mask.add", json!({"kind": "prompt", "text": "sky"})), ("segment.prepare", json!({}))]
    {
        let t = Instant::now();
        let e = s.execute(cmd, &p).unwrap_err().to_string();
        assert!(t.elapsed() < Duration::from_millis(500), "{cmd}");
        if Segmenter::AVAILABLE {
            assert!(e.starts_with(NOT_INSTALLED), "{cmd}: {e}");
        } else {
            assert!(e.contains("not available"), "{cmd}: {e}");
        }
    }
    assert!(s.develop_of(id).unwrap().masks.is_empty());
    if !Segmenter::AVAILABLE {
        return;
    }
    // a model folder whose weights are damaged
    let dir = damaged_model("damaged");
    s.segmenter.dir = Some(dir.clone());
    assert!(s.segmenter.installed());
    let t = Instant::now();
    s.execute("mask.add", &json!({"kind": "object"})).unwrap();
    let r = s.execute("mask.objectPoint", &json!({"x": 0.5, "y": 0.5})).unwrap();
    assert_eq!(r["pending"], true);
    assert!(s.segmenter.pending_clicks().is_some_and(|p| p.hint.len() == 1));
    // a second click while the first is on its way builds on it
    let r = s.execute("mask.objectPoint", &json!({"x": 0.6, "y": 0.5, "exclude": true})).unwrap();
    assert_eq!((r["include"].as_u64(), r["exclude"].as_u64()), (Some(1), Some(1)));
    let r = s.execute("mask.add", &json!({"kind": "prompt", "text": "the sky"})).unwrap();
    assert_eq!(r["pending"], true);
    assert!(t.elapsed() < Duration::from_secs(1), "nothing waited for the model: {:?}", t.elapsed());
    let messages = settle(&mut s, Duration::from_secs(60));
    assert!(!messages.is_empty() && messages.iter().all(|m| !m.is_empty()), "{messages:?}");
    assert!(messages.iter().any(|m| m.contains("model.safetensors") || m.contains("SAM 3")), "{messages:?}");
    assert!(s.segmenter.pending_clicks().is_none());
    // the Object mask is there, without a selection; no Describe mask was made from nothing
    let d = s.develop_of(id).unwrap();
    assert_eq!(d.masks.len(), 1);
    assert!(matches!(&d.masks[0].components[0].shape, lightcraft_develop::MaskShape::Object { seg: None, .. }));
    assert!(!s.segmenter.busy() && !s.segmenter.loaded());
    let _ = std::fs::remove_dir_all(&dir);
}

/// CLI / MCP (waiting mode): a damaged model is an error from the command, quickly.
#[test]
fn waiting_requests_with_a_damaged_model_are_errors() {
    if !Segmenter::AVAILABLE {
        return;
    }
    let mut s = Session::with_demo();
    let id = s.active().unwrap();
    let dir = damaged_model("damaged-wait");
    s.segmenter.dir = Some(dir.clone());
    let e = s.execute("mask.add", &json!({"kind": "prompt", "text": "sky"})).unwrap_err().to_string();
    assert!(e.contains("model.safetensors") || e.contains("SAM 3"), "{e}");
    let e = s.execute("mask.add", &json!({"kind": "object", "points": [[0.5, 0.5]]})).unwrap_err().to_string();
    assert!(!e.is_empty());
    assert!(s.develop_of(id).unwrap().masks.is_empty());
    assert!(!s.segmenter.busy());
    // a stored segmentation needs no model at all (replayed or pasted masks)
    let seg = serde_json::to_value(lightcraft_develop::SegMask::from_logits(4, &[5.0; 16])).unwrap();
    s.segmenter.dir = None;
    s.execute("mask.add", &json!({"kind": "prompt", "text": "sky", "seg": seg})).unwrap();
    assert_eq!(s.develop_of(id).unwrap().masks.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Too many clicks on one Object selection are refused, not sent to the model.
#[test]
fn object_clicks_are_capped() {
    if !Segmenter::AVAILABLE {
        return;
    }
    let mut s = Session::with_demo();
    s.segmenter.background = true;
    let dir = damaged_model("clicks");
    s.segmenter.dir = Some(dir.clone());
    let id = s.active().unwrap();
    let mut d = (*s.develop_of(id).unwrap()).clone();
    let many = vec![lightcraft_geom::Point::new(0.5, 0.5); crate::segment::MAX_CLICKS];
    d.masks.push(lightcraft_develop::Mask {
        id: 1,
        components: vec![lightcraft_develop::MaskComponent {
            name: None,
            op: lightcraft_develop::MaskOp::Add,
            invert: false,
            shape: lightcraft_develop::MaskShape::Object {
                hint: many,
                exclude: vec![],
                seg: None,
                detail: vec![],
                edge: 0.0,
                matte: vec![],
                matting: false,
            },
        }],
        ..Default::default()
    });
    s.set_develop(id, d, "test").unwrap();
    s.active_mask = Some(1);
    let e = s.execute("mask.objectPoint", &json!({"x": 0.5, "y": 0.5})).unwrap_err().to_string();
    assert!(e.contains("at most"), "{e}");
    assert!(!s.segmenter.busy());
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "sam")]
#[test]
fn a_running_model_download_shows_in_activity_and_cancels() {
    use std::io::{BufRead, BufReader, Write};
    // a local mirror that sends its files slowly, so the download is still running when it is cancelled
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/sam3", l.local_addr().unwrap());
    std::thread::spawn(move || {
        for mut c in l.incoming().flatten() {
            let mut r = BufReader::new(c.try_clone().unwrap());
            let mut line = String::new();
            while r.read_line(&mut line).unwrap_or(0) > 2 {
                line.clear();
            }
            let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\nConnection: close\r\n\r\n");
            for _ in 0..250 {
                if c.write_all(&[b' '; 4000]).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    });
    let dir = tmp("download-activity");
    let mirrors = dir.with_extension("activity-mirrors.txt");
    std::fs::write(&mirrors, format!("# test mirror\n{base}\n")).unwrap();
    let mut s = Session::with_demo();
    s.segmenter.dir = Some(dir.clone());
    s.segmenter.mirrors_file = Some(mirrors.clone());
    assert_eq!(s.execute("segment.model.download", &json!({"acknowledged": true})).unwrap()["started"], true);
    let rows = s.activity.list();
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].kind, rows[0].label.as_str(), rows[0].detail.as_str()), ("download", "Downloading model", "SAM 3"));
    assert_eq!(rows[0].unit, crate::activity::Unit::Bytes);
    assert!(rows[0].cancellable);
    s.execute("activity.cancel", &json!({"id": rows[0].id})).unwrap();
    let t = Instant::now();
    while s.segmenter.download_status().running || !s.activity.list().is_empty() {
        assert!(t.elapsed() < Duration::from_secs(10), "the download never stopped");
        std::thread::sleep(Duration::from_millis(10));
    }
    let st = s.segmenter.download_status();
    assert!(st.cancelled, "the app tells a cancel by the flag (#1046): {st:?}");
    let e = st.error.unwrap_or_default();
    assert!(e.contains("cancelled"), "{e}");
    assert!(!s.segmenter.installed());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&mirrors);
}

/// Frontends decide on the error's kind, not its words (#1046): a missing model is
/// `ModelMissing(Sam3)` whether the app queues requests (background) or a command waits (CLI, MCP);
/// a build without AI masks is not a missing model (there is nothing to download).
#[test]
fn a_missing_model_is_model_missing_by_kind() {
    use crate::{AiModel, ErrorKind};
    for background in [true, false] {
        let mut s = Session::with_demo();
        s.segmenter.background = background;
        s.segmenter.dir = Some(tmp(&format!("kind-{background}")));
        for (cmd, p) in
            [("mask.add", json!({"kind": "object"})), ("mask.add", json!({"kind": "prompt", "text": "sky"})), ("segment.prepare", json!({}))]
        {
            let e = s.execute(cmd, &p).unwrap_err();
            if Segmenter::AVAILABLE {
                assert_eq!(e.kind(), ErrorKind::ModelMissing(AiModel::Sam3), "{cmd} (background {background}): {e}");
                assert!(e.to_string().starts_with(NOT_INSTALLED), "the message is unchanged: {e}");
            } else {
                assert_eq!(e.kind(), ErrorKind::Other, "{cmd}: {e}");
            }
        }
    }
}

/// A folder that looks like an installed ViTMatte-S but holds damaged weights.
fn damaged_matte(name: &str) -> std::path::PathBuf {
    let d = tmp(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("model.safetensors"), b"\x10\x00\x00\x00\x00\x00\x00\x00{not json at all}").unwrap();
    d
}

/// A 64² selection of a box in the middle, as stored logits.
fn box_json() -> serde_json::Value {
    let l: Vec<f32> = (0..64 * 64).map(|i| if (16..48).contains(&(i % 64)) && (16..48).contains(&(i / 64)) { 8.0 } else { -8.0 }).collect();
    serde_json::to_value(lightcraft_develop::SegMask::from_logits(64, &l)).unwrap()
}

/// Add a Describe mask with a stored selection (no model needed); returns the photo and mask id.
fn stored_mask(s: &mut Session) -> (lightcraft_catalog::PhotoId, u64) {
    let id = s.active().unwrap();
    let r = s.execute("mask.add", &json!({"kind": "prompt", "text": "person", "seg": box_json()})).unwrap();
    (id, r["activeMask"].as_u64().unwrap())
}

#[test]
fn matte_commands_without_the_model_are_model_missing_by_kind() {
    use crate::{AiModel, ErrorKind};
    for background in [true, false] {
        let mut s = Session::with_demo();
        s.segmenter.background = background;
        s.segmenter.matte_dir = Some(tmp(&format!("matte-kind-{background}")));
        let (id, _) = stored_mask(&mut s);
        for (cmd, p) in [("mask.matte", json!({"enabled": true})), ("mask.refineMatte", json!({}))] {
            let e = s.execute(cmd, &p).unwrap_err();
            if Segmenter::AVAILABLE {
                assert_eq!(e.kind(), ErrorKind::ModelMissing(AiModel::ViTMatte), "{cmd} (background {background}): {e}");
                assert!(e.to_string().starts_with(crate::segment::MATTE_NOT_INSTALLED), "{e}");
            } else {
                assert_eq!(e.kind(), ErrorKind::Other, "{cmd}: {e}");
            }
        }
        // refused before anything changed
        let d = s.develop_of(id).unwrap();
        assert!(matches!(&d.masks[0].components[0].shape, lightcraft_develop::MaskShape::Prompt { matting: false, .. }));
        // switching it off needs no model
        assert_eq!(s.execute("mask.matte", &json!({"enabled": false})).unwrap()["enabled"], false);
    }
}

#[test]
fn model_commands_know_both_models() {
    let mut s = Session::with_demo();
    let st = s.execute("segment.model.status", &json!({})).unwrap();
    assert_eq!(st["matte"]["installed"], false);
    assert_eq!(st["matte"]["busy"], false);
    assert_eq!(st["matte"]["download"]["running"], false);
    assert!(st["matte"]["license"].as_str().unwrap().contains("Apache-2.0"));
    assert_eq!(st["matte"]["sizeBytes"], crate::segment::MATTE_MODEL_BYTES);
    // consent first, naming the model, size and licence; SAM 3 stays the default
    let e = s.execute("segment.model.download", &json!({"model": "vitmatte"})).unwrap_err().to_string();
    assert!(e.contains("ViTMatte-S") && e.contains("acknowledged") && e.contains("Apache-2.0") && e.contains("GB"), "{e}");
    let e = s.execute("segment.model.download", &json!({"model": "vitmatte", "acknowledged": "yes"})).unwrap_err().to_string();
    assert!(e.contains("acknowledged"), "{e}");
    let e = s.execute("segment.model.download", &json!({})).unwrap_err().to_string();
    assert!(e.contains("SAM 3"), "{e}");
    for cmd in ["segment.model.download", "segment.model.cancel"] {
        let e = s.execute(cmd, &json!({"model": "other", "acknowledged": true})).unwrap_err().to_string();
        assert!(e.contains("vitmatte"), "{cmd}: {e}");
    }
    assert_eq!(s.execute("segment.model.cancel", &json!({"model": "vitmatte"})).unwrap()["cancelled"], false);
    if !Segmenter::AVAILABLE {
        return;
    }
    let e = s.execute("segment.model.download", &json!({"model": "vitmatte", "acknowledged": true})).unwrap_err().to_string();
    assert!(e.contains("no folder"), "{e}");
}

#[cfg(feature = "sam")]
#[test]
fn a_running_matte_download_shows_in_activity_and_cancels() {
    use std::io::{BufRead, BufReader, Write};
    // a local mirror that sends slowly, so the download is still running when it is cancelled
    // (it is the only mirror that is reached: cancelling ends the download, nothing falls through)
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/vitmatte", l.local_addr().unwrap());
    std::thread::spawn(move || {
        for mut c in l.incoming().flatten() {
            let mut r = BufReader::new(c.try_clone().unwrap());
            let mut line = String::new();
            while r.read_line(&mut line).unwrap_or(0) > 2 {
                line.clear();
            }
            let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 103294572\r\nConnection: close\r\n\r\n");
            for _ in 0..500 {
                if c.write_all(&[b' '; 4000]).is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    });
    let dir = tmp("matte-download-activity");
    let mirrors = dir.with_extension("mirrors.txt");
    std::fs::write(&mirrors, format!("# test mirror\n{base}\n")).unwrap();
    let mut s = Session::with_demo();
    s.segmenter.matte_dir = Some(dir.clone());
    s.segmenter.matte_mirrors_file = Some(mirrors.clone());
    assert_eq!(s.execute("segment.model.download", &json!({"model": "vitmatte", "acknowledged": true})).unwrap()["started"], true);
    assert!(s.segmenter.matte_download_status().running && !s.segmenter.download_status().running, "the SAM 3 download is a separate one");
    let rows = s.activity.list();
    assert_eq!(rows.len(), 1);
    assert_eq!((rows[0].kind, rows[0].label.as_str(), rows[0].detail.as_str()), ("download", "Downloading model", "ViTMatte-S"));
    assert!(rows[0].cancellable);
    // SAM 3's cancel does not stop it; the matte one does
    assert_eq!(s.execute("segment.model.cancel", &json!({})).unwrap()["cancelled"], false);
    assert_eq!(s.execute("segment.model.cancel", &json!({"model": "vitmatte"})).unwrap()["cancelled"], true);
    let t = Instant::now();
    while s.segmenter.matte_download_status().running || !s.activity.list().is_empty() {
        assert!(t.elapsed() < Duration::from_secs(10), "the download never stopped");
        std::thread::sleep(Duration::from_millis(10));
    }
    let st = s.segmenter.matte_download_status();
    assert!(st.cancelled && !st.finished, "{st:?}");
    assert!(!s.segmenter.matte_installed());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&mirrors);
}

/// A damaged `model.safetensors` is an error from the command (CLI/MCP) and a message in the app,
/// never a crash or a stuck row; the mask is left as it was.
#[cfg(feature = "sam")]
#[test]
fn a_damaged_matte_model_is_an_error_not_a_crash() {
    let dir = damaged_matte("matte-damaged");
    let mut s = Session::with_demo();
    s.segmenter.matte_dir = Some(dir.clone());
    assert!(s.segmenter.matte_installed());
    let (id, _) = stored_mask(&mut s);
    let e = s.execute("mask.refineMatte", &json!({})).unwrap_err().to_string();
    assert!(!e.is_empty() && !e.contains("panicked"), "{e}");
    assert!(s.activity.list().is_empty(), "the row is gone");
    assert!(!s.segmenter.matte_busy());
    let matte_len = |s: &Session| match &s.develop_of(id).unwrap().masks[0].components[0].shape {
        lightcraft_develop::MaskShape::Prompt { matte, .. } => matte.len(),
        _ => usize::MAX,
    };
    assert_eq!(matte_len(&s), 0);
    // the same in the app: the request is queued, fails on the worker and comes back as a message
    s.segmenter.background = true;
    let r = s.execute("mask.refineMatte", &json!({})).unwrap();
    assert_eq!(r["started"], true, "{r}");
    let messages = settle(&mut s, Duration::from_secs(60));
    assert!(messages.iter().any(|m| m.contains("ViTMatte")), "{messages:?}");
    assert_eq!(matte_len(&s), 0);
    assert!(s.activity.list().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A finished matting pass is applied to the mask it was computed for, and dropped when the mask
/// has changed meanwhile (new clicks, another selection) or matting was switched off.
#[cfg(feature = "sam")]
#[test]
fn a_matte_outcome_for_a_changed_mask_is_dropped() {
    use crate::segment::{Tag, worker};
    use lightcraft_develop::{MaskShape, SegMask};
    let mut s = Session::with_demo();
    let (id, mid) = stored_mask(&mut s);
    let mid = mid as u32;
    let tile = |v: f32| vec![SegMask::from_alpha_in(8, &[v; 64], [0.1, 0.1, 0.2, 0.2])];
    let shape_of = |s: &Session| s.develop_of(id).unwrap().masks.iter().find(|m| m.id == mid).unwrap().components[0].shape.clone();
    let matte_of = |s: &Session| match shape_of(s) {
        MaskShape::Prompt { matte, .. } => matte,
        _ => vec![],
    };
    let send = |s: &mut Session, shape: MaskShape, matte: Vec<SegMask>| {
        let tag = Tag::Matte { photo: id, mask: mid, comp: 0, shape: crate::segment::strip_matte(shape) };
        s.segmenter.results.0.send(worker::Outcome { tag, result: Ok(None), matte, superseded: false, notice: None }).unwrap();
        s.segment_poll()
    };
    // matting off: nothing is kept
    let before = shape_of(&s);
    assert!(!send(&mut s, before.clone(), tile(0.5)).changed);
    assert!(matte_of(&s).is_empty());
    // on: applied
    s.set_develop(id, switched(&s, id, mid, true), "test").unwrap();
    let polled = send(&mut s, before.clone(), tile(0.5));
    assert!(polled.changed, "{:?}", polled.messages);
    assert_eq!(matte_of(&s).len(), 1);
    // the selection changed meanwhile (another description): the old pass is dropped
    let mut d = (*s.develop_of(id).unwrap()).clone();
    if let MaskShape::Prompt { text, matte, .. } = &mut d.masks[0].components[0].shape {
        *text = "dog".into();
        matte.clear();
    }
    s.set_develop(id, d, "test").unwrap();
    let polled = send(&mut s, before, tile(0.9));
    assert!(!polled.changed && matte_of(&s).is_empty(), "stale pass applied");
    // a failure comes back as a message, a cancel is silent
    let tag = |s: &Session| Tag::Matte { photo: id, mask: mid, comp: 0, shape: crate::segment::strip_matte(shape_of(s)) };
    let t = tag(&s);
    s.segmenter.results.0.send(worker::Outcome { tag: t, result: Err("it broke".into()), matte: vec![], superseded: false, notice: None }).unwrap();
    assert_eq!(s.segment_poll().messages, vec!["it broke".to_string()]);
    let t = tag(&s);
    s.segmenter.results.0.send(worker::Outcome { tag: t, result: Err(worker::MATTE_CANCELLED.into()), matte: vec![], superseded: false, notice: None }).unwrap();
    assert!(s.segment_poll().messages.is_empty());
}

/// The document with matting switched `on` for the first component of mask `mid`.
#[cfg(feature = "sam")]
fn switched(s: &Session, id: lightcraft_catalog::PhotoId, mid: u32, on: bool) -> lightcraft_develop::DevelopSettings {
    let mut d = (*s.develop_of(id).unwrap()).clone();
    if let Some(m) = d.masks.iter_mut().find(|m| m.id == mid)
        && let lightcraft_develop::MaskShape::Prompt { matting, .. } = &mut m.components[0].shape
    {
        *matting = on;
    }
    d
}

/// A detail result is not rejected because a matte arrived (or matting was switched) while it ran,
/// and a new detail patch drops the matte computed from the old selection.
#[cfg(feature = "sam")]
#[test]
fn a_detail_outcome_ignores_the_matte_and_clears_it() {
    use crate::segment::{Tag, worker};
    use lightcraft_develop::{MaskShape, SegMask};
    let mut s = Session::with_demo();
    let (id, mid) = stored_mask(&mut s);
    let mid = mid as u32;
    // the pass was started from the mask as it is now ...
    let started = s.develop_of(id).unwrap().masks[0].components[0].shape.clone();
    // ... and meanwhile a matte arrived and matting was switched on
    let mut d = switched(&s, id, mid, true);
    if let MaskShape::Prompt { matte, .. } = &mut d.masks[0].components[0].shape {
        *matte = vec![SegMask::from_alpha_in(8, &[0.5; 64], [0.1, 0.1, 0.2, 0.2])];
    }
    s.set_develop(id, d, "test").unwrap();
    let patch = SegMask::from_logits_in(8, &[4.0; 64], [0.2, 0.2, 0.8, 0.8]);
    let tag = Tag::Detail { photo: id, mask: mid, comp: 0, shape: crate::segment::strip_detail(started) };
    s.segmenter.results.0.send(worker::Outcome { tag, result: Ok(Some(patch)), matte: vec![], superseded: false, notice: None }).unwrap();
    let polled = s.segment_poll();
    assert!(polled.changed, "the detail pass was rejected");
    match &s.develop_of(id).unwrap().masks[0].components[0].shape {
        MaskShape::Prompt { detail, matte, matting, .. } => {
            assert_eq!(detail.len(), 1);
            assert!(matte.is_empty(), "the old matte no longer fits the selection");
            assert!(*matting, "the switch stays on");
        }
        other => panic!("{other:?}"),
    }
}

/// With the real model (LIGHTCRAFT_VITMATTE_DIR): a pass on the demo photo stores tiles, and a
/// cancel after the first tile stops the rest.
#[cfg(feature = "sam")]
#[test]
fn matting_with_the_real_model_runs_and_cancels_between_tiles() {
    let Some(dir) = std::env::var_os("LIGHTCRAFT_VITMATTE_DIR").map(std::path::PathBuf::from).filter(|d| lightcraft_segment::is_matte_model_dir(d)) else {
        lightcraft_testkit::skip!(Model, "LIGHTCRAFT_VITMATTE_DIR is not set");
        return;
    };
    let mut s = Session::with_demo();
    s.segmenter.matte_dir = Some(dir);
    let (id, mid) = stored_mask(&mut s);
    let matte_len = |s: &Session| match &s.develop_of(id).unwrap().masks.iter().find(|m| m.id as u64 == mid).unwrap().components[0].shape {
        lightcraft_develop::MaskShape::Prompt { matte, .. } => matte.len(),
        _ => usize::MAX,
    };
    // waiting: done when the command returns
    let r = s.execute("mask.matte", &json!({"enabled": true})).unwrap();
    assert_eq!(r["started"], true, "{r}");
    let tiles = r["tiles"].as_u64().unwrap() as usize;
    assert!(tiles >= 1 && matte_len(&s) == tiles, "{tiles} vs {}", matte_len(&s));
    // background: cancel as soon as the first tile is done
    if tiles >= 2 {
        s.segmenter.background = true;
        s.execute("mask.refineMatte", &json!({})).unwrap();
        let t = Instant::now();
        loop {
            let rows = s.activity.list();
            if let Some(row) = rows.first() {
                assert_eq!(row.kind, "matte");
                if row.done >= 1 {
                    s.execute("activity.cancel", &json!({"id": row.id})).unwrap();
                    break;
                }
            }
            assert!(t.elapsed() < Duration::from_secs(120), "never started");
            std::thread::sleep(Duration::from_millis(5));
        }
        let messages = settle(&mut s, Duration::from_secs(120));
        assert!(messages.is_empty(), "a cancel is not an error: {messages:?}");
        assert!(s.activity.list().is_empty());
    }
}
