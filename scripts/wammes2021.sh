# Wammes et al. (2021)-style pair learning: online SGD (20 epochs), one job per seed.
#
#   bash scripts/wammes2021.sh                  # SLURM via submitit (configs/hydra/launcher/cpu.yaml)
#   LAUNCHER=local bash scripts/wammes2021.sh   # run sequentially on this machine
#
# Extra Hydra overrides are passed through, e.g.
#   bash scripts/wammes2021.sh stimuli.phi_in=0.8
# Any model from configs/model (default: configs/model/complete_model_linear.yaml), e.g.
#   NAME=complete_model_relu bash scripts/wammes2021.sh model=complete_model_relu
# Plot afterwards with
#   python scripts/plot_wammes2021.py data/wammes2021/online
LAUNCHER=${LAUNCHER:-cpu}
SEEDS=$(seq -s, 0 49)
NAME=${NAME:-online}

python scripts/tasks/wammes2021.py seed=${SEEDS} training.epochs=50 training.lr=10.0 training.momentum=0.5 training.batch_size=1 "$@" \
    hydra.sweep.dir=data/wammes2021/${NAME} hydra/launcher=${LAUNCHER} --multirun
