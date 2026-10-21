MODELS=("complete_model_linear" "complete_model_relu" "complete_model_tanh")
EXPERIMENTS=("wanjia2021")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu training.lr=0.1 training.momentum=0.5 --multirun
    done
done
