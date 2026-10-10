#!/bin/bash
# S0 MD stage, single state (for the first test run or to redo one failed state). Submit from s0/:
#   sbatch --partition=<p> --gres=gpu:1 --time=24:00:00 slurm/submit_one.sh <state_id>
# Environment comes from slurm/cluster.env (see cluster.env.example).
#SBATCH --job-name=s0_md_oc25_one
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/slurm_logs/%x_%j.out
#SBATCH --error=slurm/slurm_logs/%x_%j.err
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

STATE_ID="${1:?Usage: sbatch [slurm opts] slurm/submit_one.sh <state_id>}"

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
