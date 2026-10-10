#!/bin/bash
# S0 MD stage: one array task = one state_id (line N of ../structures/state_list.txt), one GPU each.
# Submit from the s0/ directory, giving the SLURM resources on the command line:
#   sbatch --partition=<p> --gres=gpu:1 --time=24:00:00 \
#          --array=1-$(wc -l < structures/state_list.txt)%16 slurm/submit_md_array.sh
# Cluster-specific environment (modules, conda env, HF token) comes from slurm/cluster.env.
# Re-submitting is safe: --restart resumes from the last 10 ps checkpoint.
#SBATCH --job-name=s0_md_oc25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/%x_%A_%a.err
#SBATCH --requeue
set -euo pipefail

S0="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${S0}/slurm/cluster.env"
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: ${ENV_FILE} missing; copy cluster.env.example to cluster.env and fill it in" >&2; exit 2; }
# shellcheck disable=SC1090
source "${ENV_FILE}"

module purge 2>/dev/null || true
[[ -n "${MD_MODULES:-}" ]] && module load ${MD_MODULES}
[[ -n "${CONDA_SH:-}" ]] && source "${CONDA_SH}" && conda activate "${CONDA_ENV}"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

STATE_LIST="${S0}/structures/state_list.txt"
STATE_ID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${STATE_LIST}")"
[[ -n "${STATE_ID}" ]] || { echo "ERROR: no state_id at line ${SLURM_ARRAY_TASK_ID} of ${STATE_LIST}" >&2; exit 1; }
echo "Array task ${SLURM_ARRAY_TASK_ID}: state_id=${STATE_ID}"

python "${S0}/run_md_oc25.py" \
    --state-id "${STATE_ID}" \
    --structures-dir "${S0}/structures" \
    --out-dir "${S0}/md" \
    --calculator fairchem \
    --model esen-sm-conserving-all-oc25 \
    --device cuda \
    --prerelax --prerelax-steps 200 \
    --temperature 298.0 \
    --timestep-fs 1.0 \
    --friction-per-fs 0.01 \
    --equil-ps 100 \
    --prod-ps 400 \
    --write-every-ps 1.0 \
    --snapshot-every-ps 50 \
    --restart
