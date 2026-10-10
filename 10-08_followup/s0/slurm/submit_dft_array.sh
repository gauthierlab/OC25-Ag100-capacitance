#!/bin/bash
# S0 DFT potentiometer: one array task = one job directory (line N of ../dft/jobs.txt) holding
# POSCAR/INCAR/KPOINTS/meta.json/POTCAR_ORDER.txt from ../make_vasp_inputs.py. Runs VASP, then the
# Henkelman Bader post-step, then removes WAVECAR/CHG. Submit from s0/:
#   sbatch --partition=<p> --nodes=1 --ntasks-per-node=<cores> --time=04:00:00 \
#          --array=1-$(wc -l < dft/jobs.txt)%32 slurm/submit_dft_array.sh
# Environment (VASP module, POTCAR path, mpirun, bader) comes from slurm/cluster.env.
#SBATCH --job-name=s0_dft_potentiometer
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/%x_%A_%a.err
set -euo pipefail

S0="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${S0}/slurm/cluster.env"
[[ -f "${ENV_FILE}" ]] || { echo "ERROR: ${ENV_FILE} missing; copy cluster.env.example to cluster.env and fill it in" >&2; exit 2; }
# shellcheck disable=SC1090
source "${ENV_FILE}"

module purge 2>/dev/null || true
[[ -n "${DFT_MODULES:-}" ]] && module load ${DFT_MODULES}
export VASP_PP_PATH
export OMP_NUM_THREADS=1

JOBS_FILE="${S0}/dft/jobs.txt"
JOB_DIR="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS_FILE}")"
[[ -n "${JOB_DIR}" ]] || { echo "ERROR: no job dir at line ${SLURM_ARRAY_TASK_ID} of ${JOBS_FILE}" >&2; exit 1; }
echo "DFT potentiometer job: ${JOB_DIR}"
cd "${JOB_DIR}"

# Skip if already complete (idempotent re-submission)
if [[ -f ACF.dat ]] && grep -q "E-fermi" OUTCAR 2>/dev/null; then echo "already done"; exit 0; fi

# POTCAR from POTCAR_ORDER.txt (species order matches POSCAR exactly)
rm -f POTCAR
while read -r sp rest; do
    [[ "${sp}" == \#* || -z "${sp}" ]] && continue
    setup="$(echo "${rest}" | sed -n 's/.*setup=\([^ \t]*\).*/\1/p')"
    cat "${VASP_PP_PATH}/potpaw_PBE/${setup}/POTCAR" >> POTCAR
done < POTCAR_ORDER.txt

grep -q '^NCORE' INCAR || echo "NCORE = ${NCORE}" >> INCAR
grep -q '^KPAR'  INCAR || echo "KPAR = 1" >> INCAR

${MPIRUN} ${VASP_EXE}

${CHGSUM_EXE} AECCAR0 AECCAR2
${BADER_EXE} CHGCAR -ref CHGCAR_sum

rm -f WAVECAR CHG
echo "Done: ${JOB_DIR}"
