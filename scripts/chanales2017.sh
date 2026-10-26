MODELS=("complete_model_linear")
EXPERIMENTS=("chanales2017")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu training.lr=0.1 training.momentum=0.5 --multirun
    done
done
