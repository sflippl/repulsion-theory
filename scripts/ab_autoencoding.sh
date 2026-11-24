# A-B pairs, one hidden layer, small init (init_scale 0.1), full-batch gradient descent,
# seeds 0-9, for two tasks:
#   ab_prediction    predictive network (A -> B)
#   ab_autoencoder   autoencoder ([A, B] -> [A, B])
# and four setups (configs/experiments/ab_<task>/<setup>.yaml):
#   relu_uncorr    ReLU, sampled vectors, corr(A) = 0, B1 = B2
#   relu_corr      ReLU, sampled vectors, corr(A) = 0.5 within and across groups, B1 = B2
#   linear_cifar   linear, CIFAR-100, A: different categories within and across groups,
#                  B: same image within group, same category across groups
#   relu_cifar     ReLU, same CIFAR-100 items
# Outputs: data/ab_<task>/<setup>/<seed>/
SEEDS=$(seq -s, 0 9)
LAUNCHER="${LAUNCHER:-cpu}"  # cpu: SLURM cluster; LAUNCHER=local bash scripts/ab_autoencoding.sh runs here

for task in ab_prediction ab_autoencoder; do
    for setup in relu_uncorr relu_corr linear_cifar relu_cifar; do
        python train.py +experiments="${task}/${setup}" seed="${SEEDS}" \
            hydra.sweep.dir="data/${task}/${setup}" hydra/launcher="${LAUNCHER}" --multirun
    done
done

# Once all jobs have finished:
#   python scripts/plot_ab_autoencoding.py          # figures/ab-autoencoding/<task>_<setup>.{pdf,png}
#                                                   # and a_similarity.{pdf,png} (A, all setups)
#   python scripts/summarize_ab_autoencoding.py     # pairmate scores, final losses
