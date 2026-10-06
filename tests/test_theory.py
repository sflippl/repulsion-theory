"""Tests for repulsion.theory (fast unit tests only -- no strict/slow
gradient-flow ground-truth assertions here; see notebooks/theory_analysis.ipynb
for exploratory theory-vs-gradient-descent comparisons)."""
import numpy as np
import pytest

from repulsion.theory.groups import (
    block_constant_matrix,
    build_group_vectors,
    build_stream_vectors,
    eigs_to_a,
    group_eigs,
    modes_to_a,
)
from repulsion.theory.mirror import p_B_A_from_z, z_from_p
from repulsion.theory.mode_solver import solve_mode, stream_sigmas
from repulsion.theory.kwta import build_kwta_lookup, kwta_interp, kwta_kernel
from repulsion.theory.pipeline import run_case
from repulsion.theory.groundtruth import simulate_mode_system


# ---------------------------------------------------------------------------
# Section 1: group eigendecomposition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a1,a2,a3,G,M", [
    (1.0, 0.3, 0.0, 3, 2),
    (1.0, 0.7, 0.1, 4, 3),
    (2.0, -0.2, 0.05, 2, 5),
])
def test_group_eigs_matches_eigvalsh(a1, a2, a3, G, M):
    C = block_constant_matrix(a1, a2, a3, G, M)
    true_eigs = np.sort(np.linalg.eigvalsh(C))

    lam1, lam2, lam3 = group_eigs(a1, a2, a3, G, M)
    expected = np.sort(np.concatenate([
        np.full(1, lam1), np.full(G - 1, lam2), np.full((M - 1) * G, lam3),
    ]))
    np.testing.assert_allclose(true_eigs, expected, atol=1e-10)


@pytest.mark.parametrize("a1,a2,a3,G,M", [
    (1.0, 0.3, 0.0, 3, 2),
    (1.0, 0.7, 0.1, 4, 3),
])
def test_eigs_to_a_roundtrip(a1, a2, a3, G, M):
    lam1, lam2, lam3 = group_eigs(a1, a2, a3, G, M)
    a1r, a2r, a3r = eigs_to_a(lam1, lam2, lam3, G, M)
    np.testing.assert_allclose([a1, a2, a3], [a1r, a2r, a3r], atol=1e-10)


def test_modes_to_a_matches_eigs_to_a_when_lam1_eq_lam2():
    lam12, lam3, M = 1.4, 0.6, 3
    a1, a2, a3 = modes_to_a(lam12, lam3, M)
    a1e, a2e, a3e = eigs_to_a(lam12, lam12, lam3, G=5, M=M)
    np.testing.assert_allclose([a1, a2, a3], [a1e, a2e, a3e], atol=1e-10)


def test_build_group_vectors_exact_gram():
    G, M, rho = 3, 2, 0.4
    vecs = build_group_vectors(rho, G, M)
    gram = vecs @ vecs.T
    expected = block_constant_matrix(1.0, rho, 0.0, G, M)
    np.testing.assert_allclose(gram, expected, atol=1e-8)


def test_build_stream_vectors_orthogonal_and_gram():
    G, M, rho_sA, rho_sB, phi = 2, 2, 0.6, -0.1, 0.7
    X, v_A, v_B = build_stream_vectors(rho_sA, rho_sB, phi, G, M)
    np.testing.assert_allclose(v_A @ v_B.T, 0.0, atol=1e-8)
    s_in = phi * rho_sA + (1 - phi) * rho_sB
    expected = block_constant_matrix(1.0, s_in, 0.0, G, M)
    np.testing.assert_allclose(X @ X.T, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Section 2: mirror variable
# ---------------------------------------------------------------------------

def test_p_B_A_L1_identity():
    z = np.array([-1.0, 0.0, 2.5])
    p, B, A = p_B_A_from_z(z, gamma=1.0, L=1)
    np.testing.assert_allclose(p, z)
    np.testing.assert_allclose(A, z)
    assert B is None


def test_p_B_A_L2_matches_closed_form_B():
    gamma = 0.7
    z = np.array([-1.2, 0.3, 2.0])
    p, B, A = p_B_A_from_z(z, gamma, L=2)
    B_closed = np.sqrt((gamma ** 2 + np.sqrt(gamma ** 4 + 4 * p ** 2)) / 2)
    np.testing.assert_allclose(B, B_closed, rtol=1e-8)
    np.testing.assert_allclose(B ** 2 - A ** 2, gamma ** 2, atol=1e-10)


def test_p_B_A_L3_A_is_cardano_root():
    gamma = 0.5
    z = np.array([-0.4, 0.1, 0.8])
    p, B, A = p_B_A_from_z(z, gamma, L=3)
    # A should satisfy A^3 + gamma^2 A - p == 0
    residual = A ** 3 + gamma ** 2 * A - p
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)
    np.testing.assert_allclose(B ** 2, gamma ** 2 + A ** 2, atol=1e-10)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_z_from_p_roundtrip(L):
    gamma = 0.6
    z = np.array([-0.3, 0.05, 0.5])
    p, _, _ = p_B_A_from_z(z, gamma, L)
    z_recovered = z_from_p(p, gamma, L)
    np.testing.assert_allclose(z_recovered, z, atol=1e-6)


# ---------------------------------------------------------------------------
# Section 3/4: mode solver
# ---------------------------------------------------------------------------

def test_solve_mode_single_stream_reduces_to_closed_form():
    sigma, eta, gamma, L, sigma_Y = 1.3, 0.8, 0.4, 2, 0.9
    ps, Bs, mu = solve_mode([sigma], [eta], [gamma], [L], sigma_Y)
    np.testing.assert_allclose(sigma * ps[0], sigma_Y, atol=1e-10)
    p_check, B_check, _ = p_B_A_from_z(eta * sigma * mu, gamma, L)
    np.testing.assert_allclose(ps[0], p_check, atol=1e-10)
    np.testing.assert_allclose(Bs[0], B_check, atol=1e-10)


def test_solve_mode_two_streams_conserves_target():
    sigmas, etas, gammas, Ls = [1.1, 0.9], [1.0, 2.0], [0.3, 0.3], [2, 3]
    sigma_Y = 1.0
    ps, Bs, mu = solve_mode(sigmas, etas, gammas, Ls, sigma_Y)
    total = sum(s * p for s, p in zip(sigmas, ps))
    np.testing.assert_allclose(total, sigma_Y, atol=1e-10)


def test_stream_sigmas_matches_group_eigs():
    rho_sA, rho_sB, phi_in, M = 0.4, -0.2, 0.6, 5
    sig12, sig3, s_in = stream_sigmas(rho_sA, rho_sB, phi_in, M)
    lam12, _, lam3 = group_eigs(1.0, s_in, 0.0, G=3, M=M)
    np.testing.assert_allclose(sig12 ** 2, lam12, atol=1e-10)
    np.testing.assert_allclose(sig3 ** 2, lam3, atol=1e-10)


# ---------------------------------------------------------------------------
# Section 6: kWTA kernel
# ---------------------------------------------------------------------------

def test_kwta_kernel_at_zero_is_alpha():
    for alpha in [0.01, 0.03, 0.1]:
        assert kwta_kernel(0.0, alpha) == pytest.approx(alpha, abs=0.01)


def test_kwta_kernel_monotonic_and_contractive():
    alpha = 0.03
    rhos = np.linspace(-0.9, 0.9, 9)
    values = [kwta_kernel(r, alpha) for r in rhos]
    assert all(v2 >= v1 - 1e-9 for v1, v2 in zip(values, values[1:]))
    for r, v in zip(rhos, values):
        if r > 0:
            assert v <= r + 1e-6


def test_kwta_lookup_matches_direct_calls():
    table = build_kwta_lookup([0.03])
    for rho in [-0.5, 0.0, 0.3, 0.6]:
        direct = kwta_kernel(rho, 0.03)
        interp = kwta_interp(rho, 0.03, table)
        assert interp == pytest.approx(direct, abs=0.02)


# ---------------------------------------------------------------------------
# Section 7/8: pipeline sweeps (smoke tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name,architecture", [
    ("case1", "A"), ("case2", "A"), ("case2", "B"),
    ("case3", "A"), ("case4", "B"), ("interactions", "B"),
])
def test_run_case_produces_finite_results(case_name, architecture):
    df = run_case(case_name, architecture)
    assert len(df) > 0
    for col in ["a1", "a2", "a3", "p_12", "p_3"]:
        assert col in df.columns
        assert np.isfinite(df[col]).all()
    assert (df["a1"] >= df["a2"] - 1e-8).all()


# ---------------------------------------------------------------------------
# Ground-truth simulator: lightweight smoke test only (no strict assertion)
# ---------------------------------------------------------------------------

def test_simulate_mode_system_smoke():
    rng = np.random.default_rng(0)
    result = simulate_mode_system(
        sigmas=[1.2, 0.9], etas=[1.0, 1.0], gammas=[0.3, 0.3], Ls=[2, 2],
        sigma_Y=1.0, n=4, lr=0.05, n_steps=300, init="exact", rng=rng,
        probe_stream=0, probe_c=(1.1, 0.5),
    )
    assert np.isfinite(result["loss"])
    assert all(np.isfinite(p) for p in result["ps"])
    assert np.isfinite(result["h_eig_12"])
    assert np.isfinite(result["h_eig_3"])
