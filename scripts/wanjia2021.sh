MODELS=("dual_stream_linear_severe" "dual_stream_linear") #("linear_network" "dual_stream_linear" "relu_network" "attention" "complete_model_fav" "complete_model_relu_fav")
EXPERIMENTS=("wanjia2021_v2")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu training.lr=0.01 --multirun
    done
done
