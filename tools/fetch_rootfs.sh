#!/usr/bin/env bash
# Fetch vllm/vllm-openai@<amd64 manifest digest> from Docker Hub without the
# Docker daemon and unpack it to ./rootfs (layer order, whiteouts applied).
# For boxes whose Docker root cannot hold the image. Run as a normal user.
set -euo pipefail
REPO=vllm/vllm-openai
MANIFEST=sha256:a485440c62099c39659a7b723636073ed726dcb41f94cf6326e16af3a1d6f6bc   # tag inkling-2-mini, linux/amd64, index sha256:c2dc759f...1449e
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; mkdir -p blobs
token() { curl -fsS --max-time 30 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | python3 -I -c 'import sys,json;print(json.load(sys.stdin)["token"])'; }
T=$(token)
curl -fsS --max-time 60 -H "Authorization: Bearer $T" -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
  "https://registry-1.docker.io/v2/$REPO/manifests/$MANIFEST" -o manifest.json
echo "$MANIFEST" | cut -d: -f2 | xargs -I{} sh -c 'echo "{}  manifest.json" | sha256sum -c -'
python3 -I -c 'import json;d=json.load(open("manifest.json"));print(d["config"]["digest"]);[print(l["digest"]) for l in d["layers"]]' > digests.txt
fetch() { # digest
  local d=$1 f=blobs/${1#sha256:}
  if [ -f "$f" ] && echo "${d#sha256:}  $f" | sha256sum -c --status -; then return 0; fi
  curl -fsSL --retry 5 --retry-delay 3 -C - -H "Authorization: Bearer $(token)" "https://registry-1.docker.io/v2/$REPO/blobs/$d" -o "$f.part" && mv "$f.part" "$f"
  echo "${d#sha256:}  $f" | sha256sum -c --status - || { echo "DIGEST MISMATCH $d"; return 1; }
}
export -f fetch token; export REPO
xargs -P 4 -I{} bash -c 'fetch {}' < digests.txt
echo "blobs: $(ls blobs | wc -l) files, $(du -sh blobs | cut -f1)"
cp "blobs/$(head -1 digests.txt | cut -d: -f2)" config.json
[ "${1:-}" = "--download-only" ] && exit 0

rm -rf rootfs; mkdir rootfs
n=0
tail -n +2 digests.txt | while read -r d; do
  n=$((n+1)); f=blobs/${d#sha256:}
  # whiteouts first: they refer to lower layers
  tar -tzf "$f" 2>/dev/null | grep -E '(^|/)\.wh\.' > wh.txt || true
  while read -r w; do
    dir=$(dirname "$w"); base=$(basename "$w")
    if [ "$base" = ".wh..wh..opq" ]; then
      [ -d "rootfs/$dir" ] && find "rootfs/$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    else
      rm -rf "rootfs/$dir/${base#.wh.}"
    fi
  done < wh.txt
  tar -xzf "$f" -C rootfs --no-same-owner --no-same-permissions --exclude='.wh.*' --delay-directory-restore 2>> extract.err || echo "layer $n ($d): tar reported errors (see extract.err)"
  find rootfs -type d ! -perm -u+w -exec chmod u+w {} + 2>/dev/null || true
done
rm -f wh.txt
echo "rootfs: $(du -sh rootfs | cut -f1); top level: $(ls rootfs | tr '\n' ' ')"
