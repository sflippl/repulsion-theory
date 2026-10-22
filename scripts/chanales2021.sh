MODELS=("complete_model_linear")
OBJECTIVES=("summed" "independent")
EXPERIMENTS=("chanales2021" "chanales2021_exp4")

for objective in "${OBJECTIVES[@]}"; do
    for model in "${MODELS[@]}"; do
        for experiment in "${EXPERIMENTS[@]}"; do
            python train.py model="${model}" +experiments="${experiment}" training.objective="${objective}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${objective}/${model}" hydra/launcher=cpu training.lr=0.1 training.momentum=0.5 --multirun
        done
    done
done

# Once all jobs have finished:
#   python scripts/plot_chanales2021.py data/chanales2021*/*/* --readout both
