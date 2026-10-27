EXTS=("png" "pdf")
EXPS=("chanales2021" "chanales2021_exp4")

for exp in "${EXPS[@]}"; do
    python scripts/plot_chanales2021.py "data/${exp}/independent/complete_model_linear/" --readout individual --ext "${EXTS[@]}"
    python scripts/plot_chanales2021.py "data/${exp}/summed/complete_model_linear/" --readout combined --ext "${EXTS[@]}"
    python scripts/plot_chanales2021.py "data/${exp}/summed/complete_model_linear/" --readout individual --ext "${EXTS[@]}"
done
