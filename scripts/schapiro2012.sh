# Schapiro et al. (2012)-style strong/weak pair learning, sweeping input/output salience:
#   (phi_in, phi_out) = (0.5, 0.5), (0.6, 0.4), ..., (1.0, 0.0), 20 seeds each, one job per run.
#
#   bash scripts/schapiro2012.sh                  # SLURM via submitit (configs/hydra/launcher/cpu.yaml)
#   LAUNCHER=local bash scripts/schapiro2012.sh   # run sequentially on this machine
#   SEEDS=0,1,2 NAME=test bash scripts/schapiro2012.sh   # fewer seeds, different output folder
#
# Two stimulus variants, each with four strong (0-3) and four weak (4-7) pairs:
#   equal  every pair has the between-pair correlation (0.3)
#   mixed  two strong and two weak pairs are more similar (0.7), the others 0.3
#
# stimuli.phi_out=null ties phi_out = 1 - phi_in, so all settings go out as one sweep per variant.
# Extra Hydra overrides are passed through (e.g. training.epochs=20, or any model from
# configs/model such as model=complete_model_relu; default complete_model_linear). Results land in
#   data/schapiro2012/${NAME}_<variant>/phi_in_<phi_in>/<seed>/results.npz
# Plot afterwards (one set of figures per variant, in figures/schapiro2012/${NAME}_<variant>) with
#   for v in equal mixed; do python scripts/plot_schapiro2012.py data/schapiro2012/${NAME}_$v; done
LAUNCHER=${LAUNCHER:-cpu}
NAME=${NAME:-online}
SEEDS=${SEEDS:-$(seq -s, 0 19)}
PHI_IN=0.5,0.6,0.7,0.8,0.9,1.0
STRONG_PAIRS="[0,1,2,3]"
declare -A PAIR_CORRS=(
    [equal]="[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]"
    [mixed]="[0.7,0.7,0.3,0.3,0.7,0.7,0.3,0.3]"
)

for variant in equal mixed; do
    python scripts/tasks/schapiro2012.py seed=${SEEDS} stimuli.phi_in=${PHI_IN} training.lr=0.1 training.momentum=0.5 \
        stimuli.phi_out=null "stimuli.pair_corrs=${PAIR_CORRS[$variant]}" "stimuli.strong_pairs=${STRONG_PAIRS}" \
        stimuli.between_corr=0.3 "$@" \
        hydra.sweep.dir=data/schapiro2012/${NAME}_${variant} 'hydra.sweep.subdir=phi_in_${stimuli.phi_in}/${seed}' \
        hydra/launcher=${LAUNCHER} --multirun
done
