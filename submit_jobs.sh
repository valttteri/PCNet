#!/bin/bash
#
# Usage:
#   bash submit_jobs.sh <target_dir> [--array <range>]
#
# Examples:
#   bash submit_jobs.sh cluster_scripts/gating_baselines
#   bash submit_jobs.sh cluster_scripts/corrector --array 42-44
#   bash submit_jobs.sh cluster_scripts/unsup      --array 42,43,44
#
# <target_dir>   Directory (or glob) containing .sh scripts to submit.
#                Defaults to "cluster_scripts_proj_*".
# --array <range> Optional SLURM array range passed to sbatch --array.
#                 Each task index is used as the random seed by the scripts.
#                 Omit to submit each script as a single non-array job (seed = 42).

TARGET_DIR=${1:-"cluster_scripts_proj_*"}
ARRAY_RANGE=""

# Parse optional --array flag
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --array)
            ARRAY_RANGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Target dir : $TARGET_DIR"
if [[ -n "$ARRAY_RANGE" ]]; then
    echo "Array range: $ARRAY_RANGE  (each index = one seed)"
else
    echo "Array range: none (single job per script, seed = SLURM_ARRAY_TASK_ID default)"
fi

# Collect scripts
shopt -s nullglob
FILES=()
while IFS= read -r -d $'\0'; do
    FILES+=("$REPLY")
done < <(find $TARGET_DIR -type f -name "*.sh" -print0)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .sh scripts found in $TARGET_DIR"
    exit 1
fi

echo "Found ${#FILES[@]} script(s)."
read -p "Submit all to SLURM? (y/n): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    for script in "${FILES[@]}"; do
        if [[ -n "$ARRAY_RANGE" ]]; then
            echo "Submitting (array $ARRAY_RANGE): $script"
            sbatch --array="$ARRAY_RANGE" "$script"
        else
            echo "Submitting: $script"
            sbatch "$script"
        fi
        sleep 0.5
    done
    echo "All jobs submitted."
else
    echo "Submission aborted."
fi  
