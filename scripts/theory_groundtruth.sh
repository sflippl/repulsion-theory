#!/usr/bin/env bash
# Dispatches gradient-descent ground-truth checks (run_theory_groundtruth.py)
# for a modest SUBSAMPLE of the Section-8 special-case grids (repulsion.theory
# pipeline) plus the denser "interactions" cross-grid, via Hydra --multirun on
# Slurm. The full closed-form sweeps (scripts/generate_theory_datasets.py) are
# cheap and run locally -- this script is only for the expensive gradient-flow
# verification runs, so it deliberately checks far fewer points than the full
# grids to keep the cluster job count and data/ directory size small.
set -euo pipefail

LAUNCHER="${LAUNCHER:-cpu}"
N_STEPS="${N_STEPS:-20000}"
OUT_ROOT="data/theory_groundtruth"

# ── Case 1: rho_sA1 x rho_sA2, free coupling, gamma -> 0 ──────────────────
for arch in A B; do
    python run_theory_groundtruth.py --multirun hydra/launcher="${LAUNCHER}" \
        case=case1 architecture="${arch}" which_stream=0,1 \
        rho_sA1=0.2,0.5,0.8 rho_sA2=0.2,0.5,0.8 \
        phi_in=1.0 phi_out=0.0 gamma1=1e-6 gamma2=1e-6 \
        n_steps="${N_STEPS}" \
        hydra.sweep.dir="${OUT_ROOT}/case1_${arch}"
done

# ── Case 2: gamma sweep (gamma1 == gamma2, looped in bash) ────────────────
for arch in A B; do
    for gamma in 1e-6 0.1 0.3 1.0; do
        python run_theory_groundtruth.py --multirun hydra/launcher="${LAUNCHER}" \
            case=case2 architecture="${arch}" which_stream=0,1 \
            rho_sA1=0.5 rho_sA2=0.5 phi_in=1.0 phi_out=0.0 \
            gamma1="${gamma}" gamma2="${gamma}" \
            eta1=1.0 eta2=1.0 \
            n_steps="${N_STEPS}" \
            hydra.sweep.dir="${OUT_ROOT}/case2_${arch}/gamma_${gamma}"
    done
done

# Architecture B eta-ratio transition (a handful of ratios at 2 gammas).
for gamma in 1e-6 0.3; do
    python run_theory_groundtruth.py --multirun hydra/launcher="${LAUNCHER}" \
        case=case2 architecture=B which_stream=0,1 \
        rho_sA1=0.5 rho_sA2=0.5 phi_in=1.0 phi_out=0.0 \
        gamma1="${gamma}" gamma2="${gamma}" eta1=1.0 \
        eta2=1,10,100,1000,100000 \
        n_steps="${N_STEPS}" \
        hydra.sweep.dir="${OUT_ROOT}/case2_eta_ratio/gamma_${gamma}"
done

# ── Case 3: phi_in x phi_out grid, rho_sB1=1, kWTA coupling ───────────────
for arch in A B; do
    python run_theory_groundtruth.py --multirun hydra/launcher="${LAUNCHER}" \
        case=case3 architecture="${arch}" which_stream=0,1 \
        rho_sA1=0.2 rho_sA2=0.2 rho_sB1=1.0 rho_sB2=1.0 \
        phi_in=0.2,0.5,1.0 phi_out=0.0,1.0 \
        gamma1=1e-6 gamma2=1e-6 \
        n_steps="${N_STEPS}" \
        hydra.sweep.dir="${OUT_ROOT}/case3_${arch}"
done

# ── Case 4: rho_sA1 x rho_sB1 grid, phi_in=phi_out=0.5 ────────────────────
for arch in A B; do
    for gamma in 1e-6 0.3; do
        python run_theory_groundtruth.py --multirun hydra/launcher="${LAUNCHER}" \
            case=case4 architecture="${arch}" which_stream=0,1 \
            rho_sA1=0.1,0.4,0.7 rho_sB1=-0.3,0.4 \
            phi_in=0.5 phi_out=0.5 gamma1="${gamma}" gamma2="${gamma}" \
            n_steps="${N_STEPS}" \
            hydra.sweep.dir="${OUT_ROOT}/case4_${arch}/gamma_${gamma}"
    done
done

# ── Interactions: denser rho x rho x gamma (architecture B only) ─────────
python run_theory_groundtruth.py --multirun hydra/launcher="${LAUNCHER}" \
    case=interactions architecture=B which_stream=0,1 \
    rho_sA1=0.1,0.5,0.9 rho_sA2=0.1,0.5,0.9 \
    gamma1=1e-6,0.3,1.0 gamma2=1e-6,0.3,1.0 \
    eta1=1.0 eta2=1,1000 \
    n_steps="${N_STEPS}" \
    hydra.sweep.dir="${OUT_ROOT}/interactions_B"

echo "Done. Aggregate with: python scripts/collect_theory_groundtruth.py ${OUT_ROOT}"
