#!/usr/bin/env bash
# mac_pipeline/run_all.sh -- the whole pipeline, unattended, on the Mac.
#
#   cd <repo>                      # the directory that contains mac_pipeline/
#   bash mac_pipeline/run_all.sh               # full rerun: every model x every dataset (~5-6 h)
#   MODE=minimal bash mac_pipeline/run_all.sh  # only what Results_Run1 lacks: csqa x 4, llama3b arc (~1 h)
#   bash mac_pipeline/run_all.sh resume        # skip steps that already finished (markers in logs/done/)
#
# Before the first run, once:
#   sudo visudo        ->  <user> ALL = (root) NOPASSWD: /usr/bin/powermetrics
#   huggingface-cli login   (Llama-3.2 is gated)
#   plug in, lid open, brightness fixed, close other apps. The script keeps the Mac and the
#   display awake with caffeinate; do not let the display sleep mid-run, it changes idle power.
#
# Everything is logged to mac_pipeline/logs/run_<timestamp>.log and one file per step.
# A failed scoring step stops the run (later steps need its output). A failed end-to-end
# configuration is logged and the others continue.

set -uo pipefail

# ------------------------------------------------------------------ settings
MODE="${MODE:-full}"                  # full | minimal
COOLDOWN="${COOLDOWN:-300}"           # seconds of idle between model runs (thermal reset)
REPEATS="${REPEATS:-3}"               # energy repeats on the 200-question MMLU subset
PYTHON="${PYTHON:-python3}"
# roles, pool and datasets come from config.py so adapter roles / new datasets need no edit here
_cfg() { "$PYTHON" -c "import sys; sys.path.insert(0, 'mac_pipeline'); import config; print(' '.join(config.$1))"; }
read -r -a ROLES    <<< "$(_cfg ALL_ROLES)"
read -r -a POOL     <<< "$(_cfg POOL)"
read -r -a DATASETS <<< "$(_cfg DATASETS)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
fi

LOGS="$HERE/logs"; DONE="$LOGS/done"; mkdir -p "$LOGS" "$DONE"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN="$LOGS/run_$STAMP.log"
RESUME=0; [[ "${1:-}" == "resume" ]] && RESUME=1

log()  { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$MAIN"; }
secs() { date +%s; }

# run one step: name, then the command. Skips if marker exists and resume mode is on.
step() {
  local name="$1"; shift
  if [[ $RESUME -eq 1 && -f "$DONE/$name" ]]; then log "skip  $name (done earlier)"; return 0; fi
  log "start $name"
  local t0; t0=$(secs)
  if "$@" >"$LOGS/${name}_$STAMP.log" 2>&1; then
    touch "$DONE/$name"
    log "ok    $name  ($(( $(secs) - t0 ))s)"
    return 0
  else
    log "FAIL  $name  ($(( $(secs) - t0 ))s)  -- see $LOGS/${name}_$STAMP.log"
    tail -n 25 "$LOGS/${name}_$STAMP.log" | sed 's/^/        /' | tee -a "$MAIN"
    return 1
  fi
}

# verify a step's outputs (files present, row counts right, energy from powermetrics)
check() {
  local name="$1"; shift
  if "$PYTHON" mac_pipeline/check_outputs.py "$@" >"$LOGS/check_${name}_$STAMP.log" 2>&1; then
    log "check $name ok"; return 0
  else
    log "CHECK FAILED $name -- see $LOGS/check_${name}_$STAMP.log"
    grep -A20 "CHECK FAILED" "$LOGS/check_${name}_$STAMP.log" | sed 's/^/        /' | tee -a "$MAIN"
    return 1
  fi
}

cooldown() {
  [[ "$COOLDOWN" -gt 0 ]] || return 0
  log "cooldown ${COOLDOWN}s"
  sleep "$COOLDOWN"
}

notify() {
  command -v osascript >/dev/null 2>&1 && \
    osascript -e "display notification \"$1\" with title \"mac_pipeline\"" >/dev/null 2>&1 || true
}

fail_hard() { log "STOPPING: $1"; notify "run stopped: $1"; exit 1; }

# ------------------------------------------------------------------ preflight
log "mode=$MODE  cooldown=${COOLDOWN}s  repeats=$REPEATS  python=$($PYTHON --version 2>&1)"
[[ "$(uname)" == "Darwin" ]] || fail_hard "this script is for the Mac"

if ! sudo -n powermetrics --samplers cpu_power -n 1 -i 100 >/dev/null 2>&1; then
  fail_hard "sudo -n powermetrics needs a password. Run 'sudo visudo' and add: $USER ALL = (root) NOPASSWD: /usr/bin/powermetrics"
fi
log "powermetrics: password-less sudo ok"

# keep machine + display awake for the whole run
caffeinate -dims -w $$ &
log "caffeinate started (pid $!)"

# pre-download every model so no scoring run waits on the network
step predownload "$PYTHON" - <<'EOF'
import sys, os
sys.path.insert(0, "mac_pipeline")
import config
from huggingface_hub import snapshot_download
for role, m in config.MODELS.items():
    p = snapshot_download(m["id"], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken"])
    print(role, m["id"], "->", p, flush=True)
snapshot_download(config.ROUTER_EMBEDDER)
print("all models cached", flush=True)
EOF
[[ $? -eq 0 ]] || fail_hard "model download failed"

# question bank (idempotent: keeps existing folds)
step 01_build_questions "$PYTHON" mac_pipeline/01_build_questions.py || fail_hard "question bank"
check 01_build_questions questions || fail_hard "question bank outputs"

# environment + energy check + prompt-format choice (writes config_overrides.json)
if [[ "$MODE" == "full" ]]; then
  step 00_check_env "$PYTHON" mac_pipeline/00_check_env.py || fail_hard "environment check"
else
  step 00_check_env_light "$PYTHON" mac_pipeline/00_check_env.py --skip-models || fail_hard "environment check"
fi
cooldown

# ------------------------------------------------------------------ scoring
if [[ "$MODE" == "full" ]]; then
  for role in "${ROLES[@]}"; do
    step "02_score_$role" "$PYTHON" mac_pipeline/02_score_model.py --role "$role" --dataset "${DATASETS[@]}" \
      || fail_hard "scoring $role"
    check "02_score_$role" scores --role "$role" --datasets "${DATASETS[@]}" || fail_hard "scores for $role incomplete"
    cooldown
  done
  for k in $(seq 1 "$REPEATS"); do
    for role in "${ROLES[@]}"; do
      step "02_repeat${k}_$role" "$PYTHON" mac_pipeline/02_score_model.py --role "$role" --dataset mmlu --limit 200 --repeat "$k" \
        || log "repeat $k for $role failed; continuing"
      check "02_repeat${k}_$role" scores --role "$role" --datasets mmlu --repeat "$k" --limit 200 || log "repeat $k for $role incomplete; continuing"
      cooldown
    done
  done
else
  # minimal: the pieces Results_Run1 does not have. Existing scores/*.csv must already be in
  # mac_pipeline/scores/ (copy them from Results_Run1 first).
  for role in base qwen3b phi4mini; do
    [[ -f "mac_pipeline/scores/$role.csv" ]] || fail_hard "minimal mode needs mac_pipeline/scores/$role.csv from Results_Run1"
    step "02_score_${role}_csqa" "$PYTHON" mac_pipeline/02_score_model.py --role "$role" --dataset csqa \
      || fail_hard "scoring $role csqa"
    check "02_score_${role}_csqa" scores --role "$role" --datasets csqa || fail_hard "csqa scores for $role incomplete"
    cooldown
  done
  step "02_score_llama3b_csqa_arc" "$PYTHON" mac_pipeline/02_score_model.py --role llama3b --dataset csqa arc \
    || fail_hard "scoring llama3b"
  check "02_score_llama3b_csqa_arc" scores --role llama3b --datasets csqa arc || fail_hard "llama3b scores incomplete"
  # every role must now cover every dataset before the analysis steps
  for role in "${ROLES[@]}"; do
    check "02_coverage_$role" scores --role "$role" --datasets "${DATASETS[@]}" || fail_hard "$role does not cover every dataset"
  done
  cooldown
fi

# ------------------------------------------------------------------ analysis (seconds)
step 03_gate         "$PYTHON" mac_pipeline/03_gate.py         || fail_hard "gate"
check 03_gate gate                                              || fail_hard "gate outputs"
step 04_train_router "$PYTHON" mac_pipeline/04_train_router.py || fail_hard "router"
check 04_train_router router                                    || fail_hard "router outputs"
step 05_simulate     "$PYTHON" mac_pipeline/05_simulate.py     || fail_hard "simulate"
check 05_simulate simulate                                      || fail_hard "simulate outputs"

# ------------------------------------------------------------------ end-to-end (measured)
# every configuration, so the paper can pick: hybrid, each plain cascade, the base alone,
# each pool model alone. A failure here does not stop the others.
E2E=(hybrid)
for p in "${POOL[@]}"; do E2E+=("cascade:$p"); done
E2E+=(base)
for p in "${POOL[@]}"; do E2E+=("single:$p"); done
TAU_STAR="$("$PYTHON" -c "import json;print(f\"{json.load(open('mac_pipeline/router/thresholds.json'))['tau_star']:.2f}\")")"
for cfg in "${E2E[@]}"; do
  name="06_e2e_${cfg//:/-}"
  case "$cfg" in base) tau="0.00";; single:*) tau="1.01";; *) tau="$TAU_STAR";; esac
  step "$name" "$PYTHON" mac_pipeline/06_run_end_to_end.py --config "$cfg" || { log "end-to-end $cfg failed; continuing"; cooldown; continue; }
  check "$name" e2e --name "${cfg//:/-}_$tau" || log "end-to-end $cfg outputs incomplete; continuing"
  cooldown
done

# ------------------------------------------------------------------ report
step 07_report "$PYTHON" mac_pipeline/07_report.py || log "report failed"
check 07_report report || log "report outputs incomplete"

# ------------------------------------------------------------------ summary
log "finished. results in mac_pipeline/results/ (report.md, results.xlsx); logs in $LOGS"
{
  echo; echo "=== step outcomes ==="
  grep -E '^\S+ \S+  (ok|FAIL|skip) ' "$MAIN"
  echo; echo "=== gate decision ==="; sed -n '/## Decision/,/## Per dataset/p' mac_pipeline/results/gate.md 2>/dev/null | head -5
  echo; echo "=== Table 3 ==="; sed -n '/## Table 3/,/## Table 4/p' mac_pipeline/results/report.md 2>/dev/null
} | tee -a "$MAIN"
notify "pipeline finished -- see mac_pipeline/results/report.md"
