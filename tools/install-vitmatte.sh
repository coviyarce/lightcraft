#!/bin/sh
# Install the ViTMatte-S model for LightCraft's AI matting (see docs/ai-masks.md).
#
# The app offers to download the model itself; this script is for developers (reference tests,
# benchmarks) and for installing from Hugging Face directly. The weights are not part of
# LightCraft: they are the hustvl/vitmatte-small-composition-1k checkpoint (Apache-2.0), which is
# public, so no token is needed or sent. The download is checked against the pinned size and SHA-256.
#
# Options:
#     --dir DIR     where to install (default: the folder LightCraft looks in; or LIGHTCRAFT_VITMATTE_DIR)
#     --check       only verify an existing installation
set -eu

REVISION="6a58ad7646403c1df626fbd746900aec7361ea1d"
URL="https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/$REVISION/model.safetensors"
SHA256="bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb"
SIZE=103294572
CHECK_ONLY=0

case "$(uname -s)" in
    Darwin) DEFAULT_DIR="$HOME/Library/Application Support/LightCraft/models/vitmatte-s" ;;
    *) DEFAULT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/lightcraft/models/vitmatte-s" ;;
esac
DIR="${LIGHTCRAFT_VITMATTE_DIR:-$DEFAULT_DIR}"

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) DIR="$2"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

sha256() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else sha256sum "$1" | cut -d' ' -f1; fi
}

verify() {
    [ -s "$DIR/model.safetensors" ] || { echo "missing: $DIR/model.safetensors" >&2; return 1; }
    got=$(sha256 "$DIR/model.safetensors")
    if [ "$got" != "$SHA256" ]; then
        echo "model.safetensors does not match the official checkpoint (got $got)" >&2
        return 1
    fi
    echo "ViTMatte-S is installed in $DIR"
}

if [ "$CHECK_ONLY" = 1 ]; then verify; exit $?; fi

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
mkdir -p "$DIR"

if [ -s "$DIR/model.safetensors" ] && [ "$(sha256 "$DIR/model.safetensors")" = "$SHA256" ]; then
    echo "model.safetensors already installed"
else
    echo "fetching model.safetensors (103 MB; resumes if interrupted)"
    part="$DIR/model.safetensors.part"
    # a wrong model.safetensors is never resumed from: it goes
    rm -f "$DIR/model.safetensors"
    if [ -f "$part" ] && [ "$(wc -c < "$part" | tr -d ' ')" -ge "$SIZE" ]; then rm -f "$part"; fi
    code=$(curl --proto '=https' -fsSL -C - -w '%{http_code}' -o "$part" "$URL") || true
    case "$code" in
        200|206) ;;
        416) rm -f "$part"; echo "the partial download doesn't match the server's; run the script again" >&2; exit 1 ;;
        *) echo "download failed (${code:-no response})" >&2; exit 1 ;;
    esac
    have=$(wc -c < "$part" | tr -d ' ')
    if [ "$have" -lt "$SIZE" ]; then
        echo "incomplete download ($have of $SIZE bytes); run the script again to resume" >&2
        exit 1
    fi
    if [ "$have" != "$SIZE" ] || [ "$(sha256 "$part")" != "$SHA256" ]; then
        rm -f "$part"
        echo "the download is damaged or not the official checkpoint; deleted it, run the script again" >&2
        exit 1
    fi
    mv -f "$part" "$DIR/model.safetensors"
fi

verify
