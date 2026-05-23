#!/usr/bin/env bash
set -euo pipefail

logdir=""
timeout=120
interval=2

usage() {
  echo "Usage:"
  echo "  $0 [--logdir DIR] [--timeout SECONDS] <slurm-script-or-sbatch-args...>"
  echo
  echo "Examples:"
  echo "  $0 slurm/run_cleanvul.slurm"
  echo "  $0 --logdir 5683241_cleanvul slurm/run_cleanvul.slurm"
  echo "  $0 --timeout 300 slurm/run_cleanvul.slurm"
  echo "  $0 --gres=gpu:1 slurm/run_cleanvul.slurm"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir|-d)
      logdir="$2"
      shift 2
      ;;
    --timeout|-t)
      timeout="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

submit_output=$(sbatch "$@")
echo "$submit_output"

jobid=$(echo "$submit_output" | grep -oE '[0-9]+' | head -n1)

if [[ -z "$jobid" ]]; then
  echo "Could not extract job ID from sbatch output:"
  echo "$submit_output"
  exit 1
fi

if [[ -z "$logdir" ]]; then
  echo "Waiting for log directory matching: ${jobid}_*"

  end=$((SECONDS + timeout))

  while true; do
    logdir=$(find . -maxdepth 1 -type d -name "${jobid}_*" -print -quit)

    if [[ -n "$logdir" ]]; then
      break
    fi

    if (( SECONDS >= end )); then
      echo "Could not find log directory for job $jobid after ${timeout}s"
      exit 1
    fi

    sleep "$interval"
  done
else
  echo "Using provided log directory: $logdir"

  end=$((SECONDS + timeout))

  while [[ ! -d "$logdir" ]]; do
    if (( SECONDS >= end )); then
      echo "Provided log directory does not exist after ${timeout}s: $logdir"
      exit 1
    fi

    sleep "$interval"
  done
fi

echo "Tailing logs from: $logdir"

tail -F "$logdir/log.out" "$logdir/log.err"