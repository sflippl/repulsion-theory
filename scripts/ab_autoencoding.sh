# A-B pairs, one hidden layer, small init (init_scale 0.1), full-batch gradient descent,
# seeds 0-9, for two tasks:
#   ab_prediction    predictive network (A -> B)
#   ab_autoencoder   autoencoder ([A, B] -> [A, B])
# and eight setups (configs/experiments/ab_<task>/<setup>.yaml):
#   relu_uncorr    ReLU, sampled vectors, corr(A) = 0, B1 = B2
#   relu_corr      ReLU, sampled vectors, corr(A) = 0.5 within and across groups, B1 = B2
#   linear_cifar   linear, CIFAR-100, A: different categories within and across groups,
#                  B: same image within group, same category across groups
#   relu_cifar     ReLU, same CIFAR-100 items
#   <setup>_behavioral   for the behavioral repulsion score (B read out from A alone): B differs
#                  within a group -- vectors corr(B1, B2) = 0.9; CIFAR B: the same image, the
#                  second rotated by 90 degrees, different categories across groups, and A: one
#                  category for all items
# Outputs: data/ab_<task>/<setup>/<seed>/
#
# Usage:
#   bash scripts/ab_autoencoding.sh [--launcher NAME] [--task TASK ...] [SETUP ...]
#     --launcher NAME   Hydra launcher: cpu (SLURM cluster, default) or local
#     --task TASK       ab_prediction and/or ab_autoencoder (repeatable; default both)
#     SETUP ...         setups to run (default all eight)
#   e.g. bash scripts/ab_autoencoding.sh relu_uncorr_behavioral relu_corr_behavioral \
#            linear_cifar_behavioral relu_cifar_behavioral
set -euo pipefail

launcher=cpu
tasks=()
setups=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --launcher) launcher="$2"; shift 2 ;;
        --task) tasks+=("$2"); shift 2 ;;
        -h|--help) sed -n '1,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        *) setups+=("$1"); shift ;;
    esac
done
[[ ${#tasks[@]} -eq 0 ]] && tasks=(ab_prediction ab_autoencoder)
[[ ${#setups[@]} -eq 0 ]] && setups=(relu_uncorr relu_corr linear_cifar relu_cifar
                                     relu_uncorr_behavioral relu_corr_behavioral
                                     linear_cifar_behavioral relu_cifar_behavioral)
seeds=$(seq -s, 0 9)

for task in "${tasks[@]}"; do
    for setup in "${setups[@]}"; do
        python train.py +experiments="${task}/${setup}" seed="${seeds}" \
            hydra.sweep.dir="data/${task}/${setup}" hydra/launcher="${launcher}" --multirun
    done
done

# Once all jobs have finished:
#   python scripts/plot_ab_autoencoding.py          # figures/ab-autoencoding/<task>_<setup>.{pdf,png},
#                                                   # a_similarity.{pdf,png} (A, all setups) and
#                                                   # behavioral_repulsion.{pdf,png}
#   python scripts/summarize_ab_autoencoding.py     # pairmate and behavioral scores, final losses
