MODELS=("complete_model_linear")
OBJECTIVES=("summed" "independent")
EXPERIMENT="chanales2021"

for objective in "${OBJECTIVES[@]}"; do
    for model in "${MODELS[@]}"; do
        python train.py model="${model}" +experiments="${EXPERIMENT}" training.objective="${objective}" seed=$(seq -s, 0 39) hydra.sweep.dir="data/${EXPERIMENT}/${objective}/${model}" hydra/launcher=cpu training.lr=0.1 training.momentum=0.5 --multirun
    done
done

# Once all jobs have finished:
#   python scripts/plot_favila2016.py data/favila2016/*/* --readout both
