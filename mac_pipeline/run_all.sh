#!/usr/bin/env bash
# mac_pipeline/run_all.sh -- the whole pipeline, unattended, on the Mac.
#
#   cd <repo>                      # the directory that contains mac_pipeline/
#   bash mac_pipeline/run_all.sh               # full rerun: every model x every dataset (~11 h, see below)
#   bash mac_pipeline/run_all.sh resume        # skip steps that already finished (markers in logs/done/)
#
# Running in sessions (to let the machine cool): Ctrl-C at any time, later run with `resume`.
# Only the step that was interrupted is redone; every finished step has a marker in logs/done/.
# Decision rule: router picks generalist or math (1.5B); confidence below the route's tau ->
# Qwen 3B. Datasets mmlu, csqa, arc (letter) + aqua (reason-then-letter, ~25-50 s/question).
# Estimated from Results_Run2 timings; the longest single step is one model's scoring (~5 h
# for the 3B because of AQuA reasoning). Check the first AQuA log for the real s/question.
#   session 1  00 (seconds) + scoring x 3 models (~9.5 h) + cooldowns                         ~10 h
#   session 2  gate/router/simulate (minutes) + 4 end-to-end configurations (~5 h) + cooldowns ~5.5 h
# E2E="route-cascade base" bash mac_pipeline/run_all.sh resume
#   limits the end-to-end step to those configurations
#   (default: base single:qwen3b cascade:qwen3b route-cascade; also available: route, single:math).
# COOLDOWN=0 disables the 5 min idle sleep between model runs.
# CHECK_MODELS=1 makes 00 re-score 200 MMLU questions per model in raw and chat format to
#   re-choose the prompt format (~1 h). Off by default: the choices from Results_Run2 are in
#   config.py; turn it on only after changing a model or the prompt.
# REPEATS=3 adds back-to-back energy repeats on 200 MMLU questions (~1.3 h). Off by default:
#   Results_Run2 showed back-to-back repeats agree to 1% while runs hours apart differ 30-44%,
#   so the between-run spread is taken from the end-to-end passes instead (07, Table 4).
#
# Before the first run, once:
#   sudo visudo        ->  <user> ALL = (root) NOPASSWD: /usr/bin/powermetrics
#   plug in, lid open, brightness fixed, close other apps. The script keeps the Mac and the
#   display awake with caffeinate; do not let the display sleep mid-run, it changes idle power.
#
# Everything is logged to mac_pipeline/logs/run_<timestamp>.log and one file per step.
# A failed scoring step stops the run (later steps need its output). A failed end-to-end
# configuration is logged and the others continue.

set -uo pipefail

# ------------------------------------------------------------------ settings
COOLDOWN="${COOLDOWN:-300}"           # seconds of idle between model runs (thermal reset)
REPEATS="${REPEATS:-0}"               # back-to-back energy repeats on 200 MMLU questions (0 = none)
CHECK_MODELS="${CHECK_MODELS:-0}"     # 1 = re-choose raw vs chat prompt per model in 00 (~1 h)
PYTHON="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
fi

# roles, pool and datasets come from config.py so new roles / datasets need no edit here.
# Read after cd + venv so the import works from any calling directory and any python.
_cfg() { "$PYTHON" -c "import sys; sys.path.insert(0, 'mac_pipeline'); import config; print(' '.join(config.$1))"; }
read -r -a ROLES    <<< "$(_cfg ALL_ROLES)"
read -r -a POOL     <<< "$(_cfg POOL)"
read -r -a DATASETS <<< "$(_cfg DATASETS)"
[[ ${#ROLES[@]} -gt 0 && ${#DATASETS[@]} -gt 0 ]] || { echo "could not read roles/datasets from mac_pipeline/config.py" >&2; exit 1; }

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
log "cooldown=${COOLDOWN}s  repeats=$REPEATS  check_models=$CHECK_MODELS  python=$($PYTHON --version 2>&1)"
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

# environment + energy check (seconds); with CHECK_MODELS=1 also the prompt-format choice
# per model on 200 MMLU questions (writes config_overrides.json, ~1 h)
if [[ "$CHECK_MODELS" == "1" ]]; then
  step 00_check_env "$PYTHON" mac_pipeline/00_check_env.py || fail_hard "environment check"
  cooldown
else
  step 00_check_env_light "$PYTHON" mac_pipeline/00_check_env.py --skip-models || fail_hard "environment check"
fi

# ------------------------------------------------------------------ scoring
# one step per model over every dataset (~40 min base, ~75 min per 3B in Results_Run2)
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

# ------------------------------------------------------------------ analysis (seconds)
step 03_gate         "$PYTHON" mac_pipeline/03_gate.py         || fail_hard "gate"
check 03_gate gate                                              || fail_hard "gate outputs"
step 04_train_router "$PYTHON" mac_pipeline/04_train_router.py || fail_hard "router"
check 04_train_router router                                    || fail_hard "router outputs"
step 05_simulate     "$PYTHON" mac_pipeline/05_simulate.py     || fail_hard "simulate"
check 05_simulate simulate                                      || fail_hard "simulate outputs"

# ------------------------------------------------------------------ end-to-end (measured)
# the configurations the paper reports; the simulation (05) covers the rest from the scoring
# data. A failure here does not stop the others.
if [[ -n "${E2E:-}" ]]; then
  read -r -a E2E <<< "$E2E"
else
  E2E=(base "single:${POOL[0]}" "cascade:${POOL[0]}" route-cascade)
fi
for cfg in "${E2E[@]}"; do
  name="06_e2e_${cfg//:/-}"
  step "$name" "$PYTHON" mac_pipeline/06_run_end_to_end.py --config "$cfg" || { log "end-to-end $cfg failed; continuing"; cooldown; continue; }
  check "$name" e2e --name "${cfg//:/-}" || log "end-to-end $cfg outputs incomplete; continuing"
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
