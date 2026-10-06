"""Section 8 special-case parameter grids and the sweep driver.

Two architectures throughout: A = (L1, L2) = (2, 2); B = (L1, L2) = (2, 3).
G = M = 2 (N = 4) throughout, matching the spec. gamma -> 0 is approximated
by a small positive proxy (1e-6).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from repulsion.theory.kwta import build_kwta_lookup, kwta_interp
from repulsion.theory.representation import multistream_repulsion

G_DEFAULT = 2
M_DEFAULT = 2
GAMMA_ZERO = 1e-6

ARCHITECTURES = {"A": [2, 2], "B": [2, 3]}
ALPHAS = [0.003, 0.01, 0.03, 0.1]

_KWTA_TABLE = None


def _kwta_table():
    global _KWTA_TABLE
    if _KWTA_TABLE is None:
        _KWTA_TABLE = build_kwta_lookup(ALPHAS)
    return _KWTA_TABLE


def _kwta(rho, alpha):
    return kwta_interp(rho, alpha, _kwta_table())


def _grid1d(lo, hi, step):
    n = round((hi - lo) / step)
    return [round(lo + i * step, 10) for i in range(n + 1)]


def _case1_points():
    rho_range = _grid1d(0.1, 0.9, 0.1)
    points = []
    # Free (no kWTA coupling): rho_sA(1) x rho_sA(2) independently.
    for rsA1, rsA2 in itertools.product(rho_range, rho_range):
        points.append(dict(
            coupling="free", alpha=None,
            rho_sA=[rsA1, rsA2], rho_sB=[0.0, 0.0],
            phi_in=1.0, phi_out=0.0, gammas=[GAMMA_ZERO, GAMMA_ZERO],
        ))
    # kWTA-coupled: rho_sA(2) = kwta(rho_sA(1), alpha), rho_sB(2) = kwta(0, alpha).
    for alpha in ALPHAS:
        for rsA1 in rho_range:
            rsA2 = _kwta(rsA1, alpha)
            rsB2 = _kwta(0.0, alpha)
            points.append(dict(
                coupling="kwta", alpha=alpha,
                rho_sA=[rsA1, rsA2], rho_sB=[0.0, rsB2],
                phi_in=1.0, phi_out=0.0, gammas=[GAMMA_ZERO, GAMMA_ZERO],
            ))
    return points


def _case2_points():
    gamma_range = [1e-6, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    eta_ratios = [1, 3, 10, 100, 1000, 1e5, 1e7]
    rho_sA1 = 0.5
    points = []
    for gamma in gamma_range:
        points.append(dict(
            coupling="free", alpha=None, eta_ratio=1.0,
            rho_sA=[rho_sA1, rho_sA1], rho_sB=[0.0, 0.0],
            phi_in=1.0, phi_out=0.0, gammas=[gamma, gamma], etas=[1.0, 1.0],
        ))
    # Architecture-B-only eta-ratio sweep (applied only when architecture == "B").
    for gamma in gamma_range:
        for ratio in eta_ratios:
            points.append(dict(
                coupling="free", alpha=None, eta_ratio=ratio, arch_only="B",
                rho_sA=[rho_sA1, rho_sA1], rho_sB=[0.0, 0.0],
                phi_in=1.0, phi_out=0.0, gammas=[gamma, gamma], etas=[1.0, ratio],
            ))
    return points


def _case3_points():
    phi_in_range = [0.2, 0.5, 0.8, 1.0]
    phi_out_range = [0.0, 0.5, 1.0]
    rho_sA1 = 0.2
    rho_sB1 = 1.0
    alpha = 0.03
    points = []
    for phi_in, phi_out in itertools.product(phi_in_range, phi_out_range):
        rsA2 = _kwta(rho_sA1, alpha)
        rsB2 = _kwta(rho_sB1, alpha)
        points.append(dict(
            coupling="kwta", alpha=alpha,
            rho_sA=[rho_sA1, rsA2], rho_sB=[rho_sB1, rsB2],
            phi_in=phi_in, phi_out=phi_out, gammas=[GAMMA_ZERO, GAMMA_ZERO],
        ))
    return points


def _case4_points():
    rho_sA1_range = [0.1, 0.4, 0.7]
    rho_sB1_range = [-0.3, 0.0, 0.4, 0.8]
    gamma_range = [GAMMA_ZERO, 0.3]
    alpha = 0.03
    points = []
    for rsA1, rsB1, gamma in itertools.product(rho_sA1_range, rho_sB1_range, gamma_range):
        rsA2 = _kwta(rsA1, alpha)
        rsB2 = _kwta(rsB1, alpha)
        points.append(dict(
            coupling="kwta", alpha=alpha,
            rho_sA=[rsA1, rsA2], rho_sB=[rsB1, rsB2],
            phi_in=0.5, phi_out=0.5, gammas=[gamma, gamma],
        ))
    return points


def _interactions_points():
    """Denser joint cross-grids beyond Cases 1-4 (rho x rho x gamma x eta_ratio)."""
    rho_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamma_range = [GAMMA_ZERO, 0.1, 0.3, 0.5, 1.0]
    eta_ratios = [1, 10, 100, 1000, 1e5]
    points = []
    for rsA1, rsA2, gamma in itertools.product(rho_range, rho_range, gamma_range):
        points.append(dict(
            coupling="free", alpha=None, eta_ratio=1.0,
            rho_sA=[rsA1, rsA2], rho_sB=[0.0, 0.0],
            phi_in=1.0, phi_out=0.0, gammas=[gamma, gamma], etas=[1.0, 1.0],
        ))
    for rsA1, rsA2, gamma, ratio in itertools.product(rho_range, rho_range, gamma_range, eta_ratios):
        points.append(dict(
            coupling="free", alpha=None, eta_ratio=ratio, arch_only="B",
            rho_sA=[rsA1, rsA2], rho_sB=[0.0, 0.0],
            phi_in=1.0, phi_out=0.0, gammas=[gamma, gamma], etas=[1.0, ratio],
        ))
    return points


CASE_SPECS = {
    "case1": _case1_points,
    "case2": _case2_points,
    "case3": _case3_points,
    "case4": _case4_points,
    "interactions": _interactions_points,
}


def run_case(case_name: str, architecture: str, G: int = G_DEFAULT, M: int = M_DEFAULT) -> pd.DataFrame:
    """Sweep `case_name`'s grid for `architecture` ("A" or "B") via the
    closed-form pipeline, reporting a1/a2/a3/p_j for both which_stream=0,1.
    """
    if case_name not in CASE_SPECS:
        raise ValueError(f"unknown case: {case_name!r}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture!r}")
    Ls = ARCHITECTURES[architecture]

    rows = []
    for point in CASE_SPECS[case_name]():
        if point.get("arch_only") is not None and point["arch_only"] != architecture:
            continue
        etas = point.get("etas", [1.0] * len(Ls))
        for which_stream in range(len(Ls)):
            result = multistream_repulsion(
                rho_sA=point["rho_sA"], rho_sB=point["rho_sB"],
                phi_in=point["phi_in"], phi_out=point["phi_out"],
                gammas=point["gammas"], etas=etas, Ls=Ls, G=G, M=M,
                which_stream=which_stream,
            )
            row = {
                "case": case_name, "architecture": architecture,
                "which_stream": which_stream,
                "rho_sA1": point["rho_sA"][0], "rho_sA2": point["rho_sA"][1],
                "rho_sB1": point["rho_sB"][0], "rho_sB2": point["rho_sB"][1],
                "phi_in": point["phi_in"], "phi_out": point["phi_out"],
                "gamma1": point["gammas"][0], "gamma2": point["gammas"][1],
                "eta1": etas[0], "eta2": etas[1],
                "eta_ratio": point.get("eta_ratio", etas[1] / etas[0] if etas[0] else np.nan),
                "coupling": point["coupling"], "alpha": point["alpha"],
            }
            if result is None:
                row.update({"a1": np.nan, "a2": np.nan, "a3": np.nan,
                            "B_12": np.nan, "B_3": np.nan, "p_12": np.nan, "p_3": np.nan})
            else:
                row.update(result)
            rows.append(row)
    return pd.DataFrame(rows)
