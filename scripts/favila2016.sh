MODELS=("complete_model_extreme_fav_v2") #("dual_stream_linear_severe" "linear_network" "dual_stream_linear" "relu_network" "attention" "complete_model_fav" "complete_model_relu_fav", "complete_model_extreme_fav" "dual_stream_relu" "dual_stream_relu_severe")
EXPERIMENTS=("favila2016" "favila2016_v2" "favila2016_nopred" "favila2016_nopred_v2" "favila2016_v3" "favila2016_nopred_v3")

for model in "${MODELS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        python train.py model="${model}" +experiments="${experiment}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${experiment}/${model}/${seed}" hydra/launcher=cpu training.lr=0.015 --multirun
    done
done
