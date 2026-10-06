"""run_theory_groundtruth.py — gradient-descent ground-truth check for one
(case, architecture, grid-point) combination of the multi-stream repulsion
theory (repulsion.theory).

For each run, computes the theoretical a1/a2/a3 (Section 7 closed-form
pipeline) and an empirical estimate from a gradient-descent simulation of the
reduced multi-stream system (Section 3, with the Section-5 representation
probe), for both "exact" (theory-idealized) and "random" weight
initializations, and writes both to result.json for later aggregation.

Usage
-----
    python run_theory_groundtruth.py case=case1 rho_sA1=0.5 rho_sA2=0.3

Multi-run sweep on the cluster:
    python run_theory_groundtruth.py --multirun hydra/launcher=cpu \\
        rho_sA1=0.1,0.3,0.5,0.7,0.9 rho_sA2=0.1,0.5,0.9

Hydra config path: configs/  (relative to this file)
Top-level config : configs/theory_groundtruth.yaml
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="theory_groundtruth", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.output_dir is not None:
        output_dir = str(cfg.output_dir)
    else:
        output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    from repulsion.theory.groundtruth import simulate_mode_system
    from repulsion.theory.mode_solver import stream_sigmas, target_sigmas
    from repulsion.theory.representation import cross_term_c12_c3, multistream_repulsion

    Ls = [2, 2] if cfg.architecture == "A" else [2, 3]
    rho_sA = [cfg.rho_sA1, cfg.rho_sA2]
    rho_sB = [cfg.rho_sB1, cfg.rho_sB2]
    gammas = [cfg.gamma1, cfg.gamma2]
    etas = [cfg.eta1, cfg.eta2]
    j = int(cfg.which_stream)

    theory = multistream_repulsion(
        rho_sA=rho_sA, rho_sB=rho_sB, phi_in=cfg.phi_in, phi_out=cfg.phi_out,
        gammas=gammas, etas=etas, Ls=Ls, G=cfg.G, M=cfg.M, which_stream=j,
    )

    result = {
        "case": cfg.case, "architecture": cfg.architecture, "which_stream": j,
        "rho_sA1": cfg.rho_sA1, "rho_sA2": cfg.rho_sA2,
        "rho_sB1": cfg.rho_sB1, "rho_sB2": cfg.rho_sB2,
        "phi_in": cfg.phi_in, "phi_out": cfg.phi_out,
        "gamma1": cfg.gamma1, "gamma2": cfg.gamma2,
        "eta1": cfg.eta1, "eta2": cfg.eta2, "alpha": cfg.alpha,
        "theory": theory,
    }

    if theory is not None:
        sig12 = [stream_sigmas(rho_sA[i], rho_sB[i], cfg.phi_in, cfg.M)[0] for i in range(2)]
        sig3 = [stream_sigmas(rho_sA[i], rho_sB[i], cfg.phi_in, cfg.M)[1] for i in range(2)]
        sigY12, sigY3, _ = target_sigmas(rho_sA[0], rho_sB[0], cfg.phi_out, cfg.M)
        c12, c3 = cross_term_c12_c3(rho_sA[j], cfg.phi_in, cfg.M)

        for init in ("exact", "random"):
            rng = np.random.default_rng(cfg.seed)
            sim12 = simulate_mode_system(
                sigmas=sig12, etas=etas, gammas=gammas, Ls=Ls, sigma_Y=sigY12,
                n=cfg.n, lr=cfg.lr, n_steps=cfg.n_steps, init=init, rng=rng,
                random_scale=cfg.random_scale, probe_stream=j, probe_c=(c12, c3),
            )
            rng = np.random.default_rng(cfg.seed + 1)
            sim3 = simulate_mode_system(
                sigmas=sig3, etas=etas, gammas=gammas, Ls=Ls, sigma_Y=sigY3,
                n=cfg.n, lr=cfg.lr, n_steps=cfg.n_steps, init=init, rng=rng,
                random_scale=cfg.random_scale, probe_stream=j, probe_c=(c12, c3),
            )
            result[f"sim_{init}"] = {
                "p_12": sim12["ps"][j], "loss_12": sim12["loss"],
                "p_3": sim3["ps"][j], "loss_3": sim3["loss"],
                "h_eig_12": sim12["h_eig_12"], "h_eig_3": sim3["h_eig_3"],
            }

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log.info("Wrote %s", os.path.join(output_dir, "result.json"))


if __name__ == "__main__":
    main()
