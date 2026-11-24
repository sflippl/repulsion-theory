source "$(dirname "$0")/steps_per_batch.sh"
MODELS=("complete_model_linear")
EXPERIMENTS=("wanjia2021")
OBJECTIVES=("summed" "independent")

for steps in "${STEPS_PER_BATCH[@]}"; do
    steps_variant "$steps"
    for objective in "${OBJECTIVES[@]}"; do
        for model in "${MODELS[@]}"; do
            for experiment in "${EXPERIMENTS[@]}"; do
                python train.py model="${model}" +experiments="${experiment}" training.objective="${objective}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}${STEPS_SUFFIX}/${objective}/${model}" hydra/launcher=cpu "${STEPS_ARGS[@]}" --multirun
            done
        done
    done
done

# Once all jobs have finished:
#   python scripts/plot_wanjia2021.py data/wanjia2021*/* --readout both
