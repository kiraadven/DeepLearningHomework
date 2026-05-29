#!/usr/bin/env bash
# Download LFW (atulanandjha/lfwpeople) from Kaggle and extract under ./data/lfw
# Prereq: pip install kaggle, and place ~/.kaggle/kaggle.json (your API token)
set -e
mkdir -p ./data/lfw
cd ./data/lfw
echo "[INFO] Downloading LFW (atulanandjha/lfwpeople) from Kaggle ..."
kaggle datasets download -d atulanandjha/lfwpeople
echo "[INFO] Unzipping ..."
unzip -o lfwpeople.zip
# Some archives nest an inner .tgz (e.g. lfw-deepfunneled.tgz). Extract any tgz/tar.gz.
for f in *.tgz *.tar.gz; do
    [ -e "$f" ] || continue
    echo "[INFO] Extracting $f ..."
    tar -xzf "$f"
done
echo "[INFO] Done. Image count:"
find . -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l
