#!/usr/bin/env bash
# Sets up the training environment outside the repository (default ~/tune-training):
#   LLaMA-Factory (pinned commit) + PyTorch for CUDA 13 in "venv", optionally llm-compressor in "venv-quant".
# Safe to run again: existing parts are kept, missing ones are added.
#
#   bash training/einrichten.sh                  # Training
#   bash training/einrichten.sh --quantisierung  # zusätzlich die Umgebung zum Quantisieren (FP8, NVFP4)
set -euo pipefail

BASIS="${TUNE_TRAINING:-$HOME/tune-training}"
LF_URL="https://github.com/hiyouga/LLaMA-Factory.git"
LF_COMMIT="${LF_COMMIT:-885a4ca}"          # state of 17.11.2026: templates qwen3_5_nothink and glm_ocr
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
PYTHON_VERSION="3.12"
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"  # the large NVIDIA wheels time out with uv's default of 30 s
QUANT=0
[[ "${1:-}" == "--quantisierung" ]] && QUANT=1

schritt() { printf '\n== %s\n' "$*"; }

command -v uv >/dev/null || { echo "uv fehlt. Installation: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v git >/dev/null || { echo "git fehlt."; exit 1; }
mkdir -p "$BASIS/laeufe"

neue_venv() {  # $1 = folder. A Python managed by uv brings its own Python.h; the system python3 on the Spark has none.
  if [[ ! -x "$1/bin/python" ]]; then
    uv venv --python-preference only-managed --python "$PYTHON_VERSION" "$1"
  fi
}

schritt "LLaMA-Factory holen (Stand $LF_COMMIT)"
if [[ ! -d "$BASIS/LLaMA-Factory/.git" ]]; then
  git clone "$LF_URL" "$BASIS/LLaMA-Factory"
fi
if ! git -C "$BASIS/LLaMA-Factory" checkout -q "$LF_COMMIT" 2>/dev/null; then
  git -C "$BASIS/LLaMA-Factory" fetch -q origin
  git -C "$BASIS/LLaMA-Factory" checkout -q "$LF_COMMIT"
fi
git -C "$BASIS/LLaMA-Factory" log -1 --format='LLaMA-Factory %h vom %ad' --date=short

schritt "Python-Umgebung $BASIS/venv"
neue_venv "$BASIS/venv"

schritt "PyTorch für CUDA 13 ($TORCH_INDEX)"
uv pip install --python "$BASIS/venv/bin/python" torch torchvision torchaudio --index-url "$TORCH_INDEX"

schritt "LLaMA-Factory installieren"
uv pip install --python "$BASIS/venv/bin/python" -e "$BASIS/LLaMA-Factory"
# PyTorch must still be the CUDA build afterwards (a resolver could have swapped it for the CPU build from PyPI).
"$BASIS/venv/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(f"FEHLER: PyTorch {torch.__version__} sieht keine Grafikkarte.")
print(f"PyTorch {torch.__version__}, Grafikkarte: {torch.cuda.get_device_name(0)}")
PY

if [[ "$QUANT" == 1 ]]; then
  schritt "Umgebung zum Quantisieren $BASIS/venv-quant (llm-compressor)"
  neue_venv "$BASIS/venv-quant"
  uv pip install --python "$BASIS/venv-quant/bin/python" torch torchvision --index-url "$TORCH_INDEX"
  uv pip install --python "$BASIS/venv-quant/bin/python" llmcompressor pillow
  "$BASIS/venv-quant/bin/python" -c "import torch, llmcompressor; assert torch.cuda.is_available(); print('llm-compressor', llmcompressor.__version__, '– PyTorch', torch.__version__)"
fi

schritt "Prüfen"
"$BASIS/venv/bin/python" "$HIER/tune.py" pruefen
echo
echo "Fertig. Weiter mit: python3 training/tune.py trainieren --help  (oder training/README.md lesen)"
