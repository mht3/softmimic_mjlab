#!/usr/bin/env bash
# Run after the main repository's setup.sh in the training conda environment.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../../.." && pwd)"
REVISION=e706e27acc88070918308a10922b8f7174d0825e
if [[ ! -d "$WORKSPACE/ardy" ]]; then
  git clone https://github.com/nv-tlabs/ardy.git "$WORKSPACE/ardy"
  git -C "$WORKSPACE/ardy" checkout "$REVISION"
fi
if [[ "$(git -C "$WORKSPACE/ardy" rev-parse HEAD)" != "$REVISION" ]]; then
  echo "Existing ARDY checkout differs from tested revision $REVISION. Preserve local work and select this revision manually." >&2
  exit 1
fi
mkdir -p "$WORKSPACE/sonic23"
"${BASE_PYTHON:-python}" -m venv --system-site-packages "$WORKSPACE/sonic23/.venv-ardy"
ARDY_PYTHON="$WORKSPACE/sonic23/.venv-ardy/bin/python"
"$ARDY_PYTHON" -m pip install -e "$WORKSPACE/ardy[demo]" mink
"$ARDY_PYTHON" - "$WORKSPACE/ardy" <<'PY_SOURCE'
from pathlib import Path
import sys,sysconfig
(Path(sysconfig.get_paths()['purelib'])/'ardy_source_path.pth').write_text(sys.argv[1]+'\n')
PY_SOURCE
CUDA_VISIBLE_DEVICES='' "$ARDY_PYTHON" "$HERE/download_models.py"
