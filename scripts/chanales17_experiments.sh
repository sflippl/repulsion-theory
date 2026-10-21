MODELS=("complete_model_extreme_fav_normalized") #("linear_network" "dual_stream_linear" "relu_network" "attention" "complete_model" "complete_model_relu", "complete_model_extreme", "dual_stream_linear_severe", "complete_model_fav" "complete_model_extreme_fav_v2")
EXPERIMENTS=("chanales2017_sim1" "chanales2017_sim1_v2" "chanales2017_sim1_v3")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu training.lr=0.1 training.momentum=0.5 --multirun
    done
done
