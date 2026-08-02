MODELS=("complete_model_fav") #("linear_network" "dual_stream_linear" "relu_network" "attention" "complete_model" "complete_model_relu", "complete_model_extreme", "dual_stream_linear_severe")
EXPERIMENTS=("chanales2021_exp1_v3" "chanales2021_exp3_v3" "chanales2021_exp4_v3")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu --multirun
    done
done
