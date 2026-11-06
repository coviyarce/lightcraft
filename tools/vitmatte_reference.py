"""Reference tensors for the Rust ViTMatte-S port (crates/segment/tests/it/vitmatte_reference.rs).

Runs the Hugging Face transformers implementation (fp32, CPU) on two synthetic images with
trimaps (made here from a fixed seed; nothing is downloaded but the checkpoint) and writes the
inputs plus the output of every stage. Case `a` is 201 x 150, which needs padding to 224 x 160 and
resizing of the position embeddings and relative position tables; case `b` is a 512 x 512 tile.
Usage (needs torch, transformers, pillow, safetensors, numpy; see requirements-vitmatte-reference.txt):
    LIGHTCRAFT_VITMATTE_DIR=<model dir> python tools/vitmatte_reference.py ref.safetensors
The model dir holds model.safetensors (hustvl/vitmatte-small-composition-1k); the architecture
(the checkpoint's config.json) is spelled out below, so no other file is needed.
"""
import os, sys
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting, VitMatteImageProcessor
from PIL import Image

DIR = os.environ.get("LIGHTCRAFT_VITMATTE_DIR", os.path.expanduser("~/.config/lightcraft/models/vitmatte-s"))
out_path = sys.argv[1]
torch.set_grad_enabled(False)

backbone = VitDetConfig(
    hidden_size=384, image_size=512, num_attention_heads=6, num_channels=4,
    out_features=["stage12"], out_indices=[12], residual_block_indices=[2, 5, 8, 11],
    use_relative_position_embeddings=True, window_block_indices=[0, 1, 3, 4, 6, 7, 9, 10], window_size=14,
)
config = VitMatteConfig(backbone_config=backbone, convstream_hidden_sizes=[48, 96, 192],
                        fusion_hidden_sizes=[256, 128, 64, 32], hidden_size=384)
model = VitMatteForImageMatting(config).eval()
model.load_state_dict(load_file(os.path.join(DIR, "model.safetensors")), strict=True)
processor = VitMatteImageProcessor(size_divisor=32)


def scene(w, h, seed):
    """A smooth coloured background with texture, a disc-shaped subject and an unknown ring."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([
        128 + 100 * np.sin(xx / 23.0 + seed), 128 + 100 * np.sin(yy / 17.0 + 2 * seed),
        128 + 100 * np.sin((xx + yy) / 31.0),
    ], -1) + rng.normal(0, 12, (h, w, 3))
    r = np.hypot(xx - w * 0.5, yy - h * 0.5) / (min(w, h) * 0.3)
    img[r < 1.0] = img[r < 1.0] * 0.4 + np.array([170, 110, 90]) * 0.6
    img = np.clip(img, 0, 255).astype(np.uint8)
    tri = np.where(r < 0.8, 255, np.where(r > 1.25, 0, 128)).astype(np.uint8)
    return img, tri


out = {}
for name, (w, h, seed) in {"a": (201, 150, 1), "b": (512, 512, 2)}.items():
    img, tri = scene(w, h, seed)
    out[f"{name}_rgb"] = torch.from_numpy(img.copy())
    out[f"{name}_trimap"] = torch.from_numpy(tri.copy())
    px = processor(images=Image.fromarray(img), trimaps=Image.fromarray(tri), return_tensors="pt")["pixel_values"]
    out[f"{name}_pixel_values"] = px.contiguous()
    stages = {}
    hooks = []
    for i, blk in enumerate(model.decoder.fusion_blocks):
        hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: stages.__setitem__(f"fusion_{i}", o)))
    for i, c in enumerate(model.decoder.convstream.convs):
        hooks.append(c.register_forward_hook(lambda m, a, o, i=i: stages.__setitem__(f"detail_{i + 1}", o)))
    bb = model.backbone.forward_with_filtered_kwargs(px, output_hidden_states=True)
    alphas = model(pixel_values=px).alphas
    for h_ in hooks:
        h_.remove()
    for i, hs in enumerate(bb.hidden_states):
        stages[f"hidden_{i}"] = hs
    stages["alphas"] = alphas
    for k, v in stages.items():
        out[f"{name}_{k}"] = v.contiguous().float() if v.dtype.is_floating_point else v
    print(name, w, h, "padded", tuple(px.shape), "alpha range", float(alphas.min()), float(alphas.max()))

save_file(out, out_path)
print("wrote", out_path, len(out), "tensors")
