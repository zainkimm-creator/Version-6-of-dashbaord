#!/usr/bin/env bash
# Out-feeder BO campaign on the GPU: commission 6 plants in sequence, then N cell workers.
#   bash src/launch_bo_outfeeder.sh [N_WORKERS]      (log: reports/bo_outfeeder.log)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
PY=../.venv/bin/python
N="${1:-6}"
LOCK=reports/bo_outfeeder.lock
if [[ -f $LOCK ]] && ps -p "$(cat $LOCK)" >/dev/null 2>&1; then echo "already running: $(cat $LOCK)"; exit 1; fi
echo $$ > $LOCK
export LD_LIBRARY_PATH="$(ls -d "$HERE"/../.venv/lib/python3*/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "start $(date -Is) workers=$N"
# Sequential: six 6-D searches at once ran the 16 GB GPU out of memory (2026-10-07).
mkdir -p reports/bo_outfeeder_cells/stderr
for p in P001 P049 P053 P158 P186 P189; do
  $PY -c "import sys; sys.path[:0]=['src','..']; import jax, json, run_bo_outfeeder as B; B.require_gpu(jax.devices()); r=B.commission('$p'); print(json.dumps({'commissioned':'$p','kp':r['kp'],'ti':r['ti'],'S_twin':r['S_twin'],'seconds':r.get('seconds')}), flush=True)" 2>reports/bo_outfeeder_cells/stderr/commission_$p.log
done
echo "commissioning done $(date -Is)"
for i in $(seq 0 $((N-1))); do
  $PY src/run_bo_outfeeder.py --shard "$i/$N" ${PLANTS:+--plants "$PLANTS"} 2>reports/bo_outfeeder_cells/stderr/shard_$i.log &
done
wait
echo "done $(date -Is)"
rm -f $LOCK
