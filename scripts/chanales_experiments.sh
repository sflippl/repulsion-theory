MODELS=("linear_network" "dual_stream_linear" "relu_network")
EXPERIMENTS=("chanales2021_exp1" "chanales2021_exp3" "chanales2021_exp4")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu --multirun
    done
done
