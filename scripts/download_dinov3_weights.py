"""Download DINOv3 pretrained weights from Meta-signed URLs.

Each URL is valid for ~1 week from the time Meta granted access.
Re-run this script with fresh links if downloads expire.
"""

import os
import sys
import urllib.request
from pathlib import Path

CKPT_DIR = Path("/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/zehao/4dmap/4dmap_policy/checkpoints/dinov3")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# All URLs share the same Policy/Signature/Key-Pair-Id from the access grant.
SHARED_QUERY = (
    "Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiaWp6MTA3aDdqMXF0dnFibjRwNGs3dDN5IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3Nzk2MDkyOTd9fX1dfQ__"
    "&Signature=obvk-LlYKiMqgmJoFwnW-UivAJszb2JdTIqnKvF06Z5cIGOqzKOsvRkdmIzN02h0HzQfNkoRoNmZ85EVa97rwENv%7EZJbo9fPCtywkzOTXnypTs6N5sn%7EGyKBeEgGStVa6hmMc60T0gIFmN9p5zye-a7yp9O-ZffthDBKh3EImcXn4y6JXZNXxIgWcDK2JL38n%7EkHoSI0zJHIkJb6SB2q-3%7EJw0Zj1jjNbbCVDF51tz8jZy0twF7QbjlE32Wc-Fm-Jv1QC3DETWcs8JfYTb9cbv3mWkAatP%7EnoqP3OEtGPWS16ckgjx2JZCkbv0sV2PXB%7Ew6FqkMEiCsRsDPUFT0kxg__"
    "&Key-Pair-Id=K15QRJLYKIFSLZ"
    "&Download-Request-ID=1511083737096550"
)

# (model_dir, filename) — backbones only, skip 7B (~28GB) unless explicitly requested.
TARGETS = [
    ("dinov3_vits16", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    ("dinov3_vits16plus", "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
    ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    ("dinov3_vith16plus", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    ("dinov3_convnext_tiny", "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"),
    ("dinov3_convnext_small", "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"),
    ("dinov3_convnext_base", "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"),
    ("dinov3_convnext_large", "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"),
    ("dinov3_vitl16", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
]


def report_progress(blocknum, bs, total):
    if total <= 0:
        return
    pct = min(100.0, blocknum * bs * 100.0 / total)
    sys.stdout.write(f"\r  {pct:5.1f}%  ({blocknum*bs/1e6:8.1f} / {total/1e6:8.1f} MB)")
    sys.stdout.flush()


def main():
    only = set(sys.argv[1:])  # optional CLI args to filter by basename
    for model_dir, fname in TARGETS:
        if only and fname not in only:
            continue
        out = CKPT_DIR / fname
        if out.exists() and out.stat().st_size > 1024 * 1024:
            print(f"[skip] {fname}  ({out.stat().st_size/1e6:.1f} MB present)")
            continue
        url = f"https://dinov3.llamameta.net/{model_dir}/{fname}?{SHARED_QUERY}"
        print(f"[get ] {fname}")
        try:
            urllib.request.urlretrieve(url, out, reporthook=report_progress)
            print(f"\n[done] {fname} -> {out} ({out.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            print(f"\n[fail] {fname}: {e}")
            if out.exists() and out.stat().st_size < 1024 * 1024:
                out.unlink()


if __name__ == "__main__":
    main()
