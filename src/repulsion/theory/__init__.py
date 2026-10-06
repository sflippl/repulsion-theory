"""Theoretically-grounded multi-stream representational repulsion.

Implements the closed-form theory (group-structured Gram eigendecomposition,
the depth-mirror variable, the multi-stream mode solver, representation-level
a1/a2/a3 statistics, and the k-WTA output-similarity kernel), plus
gradient-descent ground-truth simulators used to cross-check the formulas,
and a pipeline for sweeping the special-case parameter grids.
"""
from repulsion.theory.groups import (
    block_constant_matrix,
    build_group_vectors,
    build_stream_vectors,
    eigs_to_a,
    group_eigs,
    modes_to_a,
)
from repulsion.theory.mirror import p_B_A_from_z, z_from_p
from repulsion.theory.mode_solver import solve_mode, stream_sigmas, target_sigmas
from repulsion.theory.representation import (
    cross_term_c12_c3,
    multistream_repulsion,
    representation_a123,
)
from repulsion.theory.kwta import build_kwta_lookup, kwta_interp, kwta_kernel
from repulsion.theory.groundtruth import (
    init_exact_weights,
    init_random_weights,
    simulate_mode_system,
)
from repulsion.theory.pipeline import CASE_SPECS, run_case

__all__ = [
    "block_constant_matrix",
    "build_group_vectors",
    "build_stream_vectors",
    "eigs_to_a",
    "group_eigs",
    "modes_to_a",
    "p_B_A_from_z",
    "z_from_p",
    "solve_mode",
    "stream_sigmas",
    "target_sigmas",
    "cross_term_c12_c3",
    "multistream_repulsion",
    "representation_a123",
    "build_kwta_lookup",
    "kwta_interp",
    "kwta_kernel",
    "init_exact_weights",
    "init_random_weights",
    "simulate_mode_system",
    "CASE_SPECS",
    "run_case",
]
