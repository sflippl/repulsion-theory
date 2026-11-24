source "$(dirname "$0")/steps_per_batch.sh"
MODELS=("complete_model_linear")
OBJECTIVES=("summed" "independent")
EXPERIMENT="favila2016"

for steps in "${STEPS_PER_BATCH[@]}"; do
    steps_variant "$steps"
    for objective in "${OBJECTIVES[@]}"; do
        for model in "${MODELS[@]}"; do
            python train.py model="${model}" +experiments="${EXPERIMENT}" training.objective="${objective}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${EXPERIMENT}${STEPS_SUFFIX}/${objective}/${model}" hydra/launcher=cpu "${STEPS_ARGS[@]}" --multirun
        done
    done
done

# Once all jobs have finished:
#   python scripts/plot_favila2016.py data/favila2016*/*/* --readout both
