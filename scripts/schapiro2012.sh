# Schapiro et al. (2012)-style strong/weak pair learning, sweeping input/output salience:
#   (phi_in, phi_out) = (0.5, 0.5), (0.6, 0.4), ..., (1.0, 0.0), 20 seeds each, one job per run.
#
#   bash scripts/schapiro2012.sh                  # SLURM via submitit (configs/hydra/launcher/cpu.yaml)
#   LAUNCHER=local bash scripts/schapiro2012.sh   # run sequentially on this machine
#   SEEDS=0,1,2 NAME=test bash scripts/schapiro2012.sh   # fewer seeds, different output folder
#
# stimuli.phi_out=null ties phi_out = 1 - phi_in, so all settings go out as one sweep.
# Extra Hydra overrides are passed through (e.g. training.epochs=20). Results land in
#   data/schapiro2012/$NAME/phi_in_<phi_in>/<seed>/results.npz
# Plot afterwards with
#   python scripts/plot_schapiro2012.py data/schapiro2012/$NAME
LAUNCHER=${LAUNCHER:-cpu}
NAME=${NAME:-online}
SEEDS=${SEEDS:-$(seq -s, 0 19)}
PHI_IN=0.5,0.6,0.7,0.8,0.9,1.0

python scripts/tasks/schapiro2012.py seed=${SEEDS} stimuli.phi_in=${PHI_IN} \
    stimuli.phi_out=null "$@" \
    hydra.sweep.dir=data/schapiro2012/${NAME} 'hydra.sweep.subdir=phi_in_${stimuli.phi_in}/${seed}' \
    hydra/launcher=${LAUNCHER} --multirun
