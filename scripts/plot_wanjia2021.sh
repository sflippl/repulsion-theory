EXTS=("png" "pdf")
EXPS=("wanjia2021")

for ext in "${EXTS[@]}"; do
    for exp in "${EXPS[@]}"; do
        python scripts/plot_wanjia2021.py "data/${exp}/independent/complete_model_linear/" --readout individual --ext "${ext}" --insight-threshold 0.6
        python scripts/plot_wanjia2021.py "data/${exp}/summed/complete_model_linear/" --readout combined --ext "${ext}" --insight-threshold 0.6
        python scripts/plot_wanjia2021.py "data/${exp}/summed/complete_model_linear/" --readout individual --ext "${ext}" --insight-threshold 0.6
    done
done