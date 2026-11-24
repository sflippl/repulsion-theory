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
# Models are any configs/model entries. STEPS_PER_BATCH (default "1 10 100") sets the optimiser
# steps per presentation; repeated steps use lr 0.05 without momentum and write to
# data/<experiment>_steps<N>/... (see scripts/steps_per_batch.sh).
source "$(dirname "$0")/steps_per_batch.sh"
LAUNCHER=${LAUNCHER:-cpu}
SEEDS=$(seq -s, 0 39)
EXPERIMENT="wammes2021${NAME:+_${NAME}}"
MODELS=("complete_model_linear")
OBJECTIVES=("summed" "independent")

for steps in "${STEPS_PER_BATCH[@]}"; do
    steps_variant "$steps"
    for objective in "${OBJECTIVES[@]}"; do
        for model in "${MODELS[@]}"; do
            python scripts/tasks/wammes2021.py seed=${SEEDS} training.epochs=10 training.batch_size=1 "${STEPS_ARGS[@]}" \
                "model=${model}" "training.objective=${objective}" "$@" \
                hydra.sweep.dir="data/${EXPERIMENT}${STEPS_SUFFIX}/${objective}/${model}" hydra/launcher=${LAUNCHER} --multirun
        done
    done
done

# Once all jobs have finished:
#   python scripts/plot_wammes2021.py data/wammes2021*/*/* --ext pdf png
