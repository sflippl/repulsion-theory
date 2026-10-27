# Wammes et al. (2021)-style pair learning: online SGD (10 epochs), one job per seed,
# for the summed (joint) and independent training objective.
#
#   bash scripts/wammes2021.sh                  # SLURM via submitit (configs/hydra/launcher/cpu.yaml)
#   LAUNCHER=local bash scripts/wammes2021.sh   # run sequentially on this machine
#
# Results land in data/<experiment>/<objective>/<model>/<seed>/results.npz, with
# experiment = wammes2021, or wammes2021_$NAME when NAME is set (for variants).
# Extra Hydra overrides are passed through, e.g.
#   NAME=phi08 bash scripts/wammes2021.sh stimuli.phi_in=0.8
# Models are any configs/model entries.
LAUNCHER=${LAUNCHER:-cpu}
SEEDS=$(seq -s, 0 39)
EXPERIMENT="wammes2021${NAME:+_${NAME}}"
MODELS=("complete_model_linear")
OBJECTIVES=("summed" "independent")

for objective in "${OBJECTIVES[@]}"; do
    for model in "${MODELS[@]}"; do
        python scripts/tasks/wammes2021.py seed=${SEEDS} training.epochs=10 training.lr=0.1 training.momentum=0.5 training.batch_size=1 \
            "model=${model}" "training.objective=${objective}" "$@" \
            hydra.sweep.dir="data/${EXPERIMENT}/${objective}/${model}" hydra/launcher=${LAUNCHER} --multirun
    done
done

# Once all jobs have finished:
#   python scripts/plot_wammes2021.py data/wammes2021/*/* --ext pdf png
