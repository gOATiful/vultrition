#!/usr/bin/env bash
set -uo pipefail

LOG_FILE="execution_times.tsv"
STOP_ON_ERROR=0   # set to 1 if you want the script to stop when a command fails

# Reset log file
printf "name\tstatus\tseconds\tcommand\n" > "$LOG_FILE"

run_timed() {
  local name="$1"
  shift

  local start end elapsed_ns elapsed status command_string
  command_string="$*"

  echo "Running: $name"

  start=$(date +%s%N)

  "$@"
  status=$?

  end=$(date +%s%N)
  elapsed_ns=$((end - start))
  elapsed=$(awk -v ns="$elapsed_ns" 'BEGIN { printf "%.3f", ns / 1000000000 }')

  printf "%s\t%s\t%s\t%s\n" "$name" "$status" "$elapsed" "$command_string" >> "$LOG_FILE"

  echo "Finished: $name took ${elapsed}s with status $status"
  echo

  if [[ "$STOP_ON_ERROR" -eq 1 && "$status" -ne 0 ]]; then
    echo "Stopping because command failed: $name"
    print_overview
    exit "$status"
  fi
}

print_overview() {
  echo
  echo "Execution time overview"
  echo "Saved to: $LOG_FILE"
  echo

  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$LOG_FILE"
  else
    cat "$LOG_FILE"
  fi
}

python3 -m pip install -e .

run_timed "BigVul" vultrition -c examples/configs/bigvul.toml --run_analysis -o vultrition_results_bigvul.json
run_timed "CleanVul" vultrition -c examples/configs/cleanvul.toml --run_analysis -o vultrition_results_cleanvul.json
run_timed "DiverseVul" vultrition -c examples/configs/diversevul.toml --run_analysis -o vultrition_results_diversevul.json
run_timed "ICVul" vultrition -c examples/configs/icvul.toml --run_analysis -o vultrition_results_icvul.json
run_timed "MegaVul_CPP" vultrition -c examples/configs/megavul_cpp.toml --run_analysis -o vultrition_results_megavul_cpp.json
run_timed "MegaVul_Java" vultrition -c examples/configs/megavul_java.toml --run_analysis -o vultrition_results_megavul_java.json
run_timed "PrimeVul" vultrition -c examples/configs/primevul.toml --run_analysis -o vultrition_results_primevul.json
run_timed "PrimeVul" vultrition -c examples/configs/primevul_paired.toml --run_analysis -o vultrition_results_primevul_paired.json
run_timed "ReposVul_C" vultrition -c examples/configs/reposvul_c.toml --run_analysis -o vultrition_results_reposvul_c.json
run_timed "ReposVul_CPP" vultrition -c examples/configs/reposvul_cpp.toml --run_analysis -o vultrition_results_reposvul_cpp.json
run_timed "ReposVul_Java" vultrition -c examples/configs/reposvul_java.toml --run_analysis -o vultrition_results_reposvul_java.json
run_timed "ReposVul_Python" vultrition -c examples/configs/reposvul_python.toml --run_analysis -o vultrition_results_reposvul_python.json
run_timed "ReposVul_Split" vultrition -c examples/configs/reposvul_split.toml --run_analysis -o vultrition_results_reposvul_split.json
run_timed "SecVulEval" vultrition -c examples/configs/secvuleval.toml --run_analysis -o vultrition_results_secvuleval.json
run_timed "TitanVul" vultrition -c examples/configs/titanvul.toml --run_analysis -o vultrition_results_titanvul.json


print_overview
