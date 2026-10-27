MODELS=("complete_model_linear")
EXPERIMENTS=("chanales2017")
OBJECTIVES=("summed" "independent")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        for objective in "${OBJECTIVES[@]}"; do
            python train.py model="${model}" +experiments="${experiment}" training.objective="${objective}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${objective}/${model}/${seed}" hydra/launcher=cpu training.lr=0.1 training.momentum=0.5 --multirun
        done
    done
done
