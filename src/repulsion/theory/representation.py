"""Representation-level a1/a2/a3 statistics (Section 5) and the full
multi-stream pipeline (Section 7) of the multi-stream repulsion spec.
"""
from __future__ import annotations

from repulsion.theory.groups import modes_to_a
from repulsion.theory.mode_solver import solve_mode, stream_sigmas, target_sigmas


def cross_term_c12_c3(rho_sA_j: float, phi_in: float, M: int):
    """a(Z^(j) X^(j)T) mode components (Section 5).

    Z^(j) is the "new-input test point" (stream j's own vectors, phi=1, pure
    A). sqrt(phi_in) scales the *whole* bracket, not just the rho_sA term.
    """
    c12 = phi_in ** 0.5 * (1.0 + (M - 1) * rho_sA_j)
    c3 = phi_in ** 0.5 * (1.0 - rho_sA_j)
    return c12, c3


def representation_a123(c12: float, c3: float, sig12_j: float, sig3_j: float,
                         B12_j: float, B3_j: float, l: int, M: int):
    """a1, a2, a3 of the last-hidden-layer representation Gram H_l H_l^T.

    `l = L_j - 1` (number of hidden layers). sig12_j/sig3_j are stream j's own
    K_X eigenvalues (already computed in Section 4 -- reused, not recomputed).
    """
    lam12_H = (c12 ** 2 / sig12_j ** 2) * B12_j ** (2 * l)
    lam3_H = (c3 ** 2 / sig3_j ** 2) * B3_j ** (2 * l)
    a1, a2, a3 = modes_to_a(lam12_H, lam3_H, M)
    return a1, a2, a3


def multistream_repulsion(rho_sA, rho_sB, phi_in: float, phi_out: float,
                           gammas, etas, Ls, G: int, M: int, which_stream: int = 0):
    """Full pipeline (Section 7).

    rho_sA, rho_sB: length-J lists, stream-specific within-group correlations.
    Y is built from stream 0's own (rho_sA[0], rho_sB[0]) with phi_out.

    Returns a dict with a1, a2, a3 of stream `which_stream`'s own
    representation at its last hidden layer, plus B_12, B_3, p_12, p_3
    (stream `which_stream`'s own p_j at mode 12 / mode 3) for diagnostics.
    Returns None if Ls[which_stream] == 1 (no hidden layer to report on).
    """
    J = len(rho_sA)
    sig12 = [None] * J
    sig3 = [None] * J
    for j in range(J):
        sig12[j], sig3[j], _ = stream_sigmas(rho_sA[j], rho_sB[j], phi_in, M)
    sigY12, sigY3, _ = target_sigmas(rho_sA[0], rho_sB[0], phi_out, M)

    ps12, Bs12, _ = solve_mode(sig12, etas, gammas, Ls, sigY12)
    ps3, Bs3, _ = solve_mode(sig3, etas, gammas, Ls, sigY3)

    j = which_stream
    if Ls[j] == 1:
        return None
    l = Ls[j] - 1
    c12, c3 = cross_term_c12_c3(rho_sA[j], phi_in, M)
    a1, a2, a3 = representation_a123(c12, c3, sig12[j], sig3[j], Bs12[j], Bs3[j], l, M)
    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "B_12": Bs12[j],
        "B_3": Bs3[j],
        "p_12": ps12[j],
        "p_3": ps3[j],
    }
