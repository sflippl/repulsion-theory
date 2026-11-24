# Sourced by the experiment sweep scripts: repeated learning on each presentation.
#
# STEPS_PER_BATCH lists the optimiser steps per presentation (batch) to run,
# default "1 10 100", e.g. STEPS_PER_BATCH="10 100" bash scripts/favila2016.sh.
# 1 is the standard setting (lr 0.1, momentum 0.5); repeated steps use lr 0.05
# without momentum and write to data/<experiment>_steps<N>/... .
read -r -a STEPS_PER_BATCH <<< "${STEPS_PER_BATCH:-1 10 100}"

# steps_variant N: set STEPS_SUFFIX and STEPS_ARGS (Hydra overrides) for N steps per presentation.
steps_variant() {
    if [ "$1" = 1 ]; then
        STEPS_SUFFIX=""
        STEPS_ARGS=(training.lr=0.1 training.momentum=0.5 training.steps_per_batch=1)
    else
        STEPS_SUFFIX="_steps$1"
        STEPS_ARGS=(training.lr=0.05 training.momentum=0 "training.steps_per_batch=$1")
    fi
}
