ext="png pdf"
EXPS=("wammes2021")

for exp in "${EXPS[@]}"; do
    python scripts/plot_wammes2021.py "data/${exp}/independent/complete_model_linear/" --readout individual --ext png pdf --epochs 2 4 10
    python scripts/plot_wammes2021.py "data/${exp}/summed/complete_model_linear/" --readout combined --ext png pdf --epochs 2 4 10
done