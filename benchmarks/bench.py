"""Benchmarks (spec section 7): timing/memory scaling, init and optimizer
comparisons, softplus vs projected G, offset vs fitted exposure, deviance vs
NMF / GLM-PCA, rotation stability on real data.

Run:  uv run --group benchmark python benchmarks/bench.py [--quick] [--out DIR]

Real dataset: 20 Newsgroups document-term counts (domain-neutral, public,
fetched via scikit-learn on first run).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from glm_seminmf import NBGLMSemiNMF, simulate_nb_seminmf

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).parent


def _fit_timed(model: NBGLMSemiNMF, X, Z=None):
    dev = model._torch_device()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(X, Z=Z)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
    else:
        peak_mb = float("nan")
    return time.perf_counter() - t0, peak_mb


def match_mean_abs_corr(F1, F2) -> float:
    from scipy.optimize import linear_sum_assignment

    k = F1.shape[1]
    C = np.nan_to_num(np.corrcoef(F1.T, F2.T)[:k, k:])  # zero-variance col -> 0
    r, c = linear_sum_assignment(-np.abs(C))
    return float(np.abs(C[r, c]).mean())


# ----------------------------------------------------------------- benchmarks


def bench_scaling(quick: bool, device: str):
    rows = []
    grid = [
        (1000, 5000, 10),
        (3000, 20000, 10),
        (3000, 20000, 30),
        (5000, 50000, 10),
    ]
    if quick:
        grid = grid[:2]
    for p, n, k in grid:
        sim = simulate_nb_seminmf(p=p, n=n, k=k, random_state=0)
        m = NBGLMSemiNMF(n_components=k, l1_F=0.001 * n, random_state=0, device=device)
        wall, mem = _fit_timed(m, sim.X)
        rows.append(
            dict(p=p, n=n, k=k, device=device, wall_s=round(wall, 1),
                 peak_gpu_mb=round(mem, 1), n_iter=m.n_iter_,
                 converged=m.converged_, dev_expl=round(m.deviance_explained_, 3))
        )
        print("scaling:", rows[-1])
    return rows


def bench_init_and_optimizer(quick: bool, device: str):
    """Convergence trace + recovery for each init and optimizer (open
    questions 1-3)."""
    p, n, k = (600, 2000, 6) if quick else (2000, 8000, 8)
    sim = simulate_nb_seminmf(p=p, n=n, k=k, random_state=1)
    lam = 0.001 * n
    out = {"traces": {}, "summary": []}

    combos = [("svd", "adam", "softplus"), ("nmf", "adam", "softplus"),
              ("random", "adam", "softplus"), ("svd", "lbfgs", "softplus"),
              ("svd", "adam_joint", "softplus"), ("svd", "adam", "projected")]
    for init, algo, gp in combos:
        m = NBGLMSemiNMF(n_components=k, l1_F=lam, init=init, algorithm=algo,
                         g_parametrization=gp, random_state=0, device=device)
        wall, _ = _fit_timed(m, sim.X)
        rec = match_mean_abs_corr(sim.F, m.F_)
        exact_zero_frac = float((m.G_ == 0).mean())
        out["traces"][f"{init}/{algo}/{gp}"] = list(m.loss_)
        out["summary"].append(
            dict(init=init, algorithm=algo, g_param=gp, wall_s=round(wall, 1),
                 n_iter=m.n_iter_, converged=m.converged_,
                 final_loss=round(m.loss_[-1], 1), recovery=round(rec, 3),
                 G_exact_zero_frac=round(exact_zero_frac, 3))
        )
        print("init/opt:", out["summary"][-1])
    return out


def bench_exposure(quick: bool, device: str):
    """Open question 5: does fitting b change recovered factors materially?"""
    p, n, k = (600, 2000, 6) if quick else (2000, 8000, 8)
    sim = simulate_nb_seminmf(p=p, n=n, k=k, random_state=2, exposure_sd=0.8)
    fits = {}
    for mode in ["offset", "fit"]:
        m = NBGLMSemiNMF(n_components=k, l1_F=0.001 * n, exposure=mode,
                         random_state=0, device=device)
        wall, _ = _fit_timed(m, sim.X)
        fits[mode] = m
        print(f"exposure={mode}: {wall:.1f}s recovery="
              f"{match_mean_abs_corr(sim.F, m.F_):.3f} dev_expl={m.deviance_explained_:.3f}")
    cross = match_mean_abs_corr(fits["offset"].F_, fits["fit"].F_)
    b_corr = float(np.corrcoef(fits["offset"].b_, fits["fit"].b_)[0, 1])
    return dict(
        recovery_offset=round(match_mean_abs_corr(sim.F, fits["offset"].F_), 3),
        recovery_fit=round(match_mean_abs_corr(sim.F, fits["fit"].F_), 3),
        cross_agreement=round(cross, 3), b_agreement=round(b_corr, 3),
    )


def load_newsgroups(quick: bool):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    docs = data.data[: 3000 if quick else 8000]
    vec = CountVectorizer(max_features=1500 if quick else 3000, min_df=5, stop_words="english")
    X = vec.fit_transform(docs).T.tocsc()  # features x samples
    keep = np.asarray(X.sum(axis=0)).ravel() >= 10  # drop near-empty docs
    return X[:, keep].astype(np.int64)


def bench_real_data(quick: bool, device: str):
    X = load_newsgroups(quick)
    p, n = X.shape
    k = 10
    lam = 0.001 * n
    print(f"20 Newsgroups: p={p} n={n} nnz_frac={X.nnz/(p*n):.4f}")
    out = {"shape": [p, n]}

    m = NBGLMSemiNMF(n_components=k, l1_F=lam, random_state=0, device=device)
    wall, mem = _fit_timed(m, X)
    dev_ours = m.deviance_explained_
    out["ours"] = dict(wall_s=round(wall, 1), dev_expl=round(dev_ours, 4),
                       n_iter=m.n_iter_, peak_gpu_mb=round(mem, 1))
    print("ours:", out["ours"])

    # rotation stability across seeds (worse on real data than synthetic)
    seeds = range(3 if quick else 5)
    fits = []
    for s in seeds:
        ms = NBGLMSemiNMF(n_components=k, l1_F=lam, random_state=s, device=device)
        ms.fit(X)
        fits.append(ms)
    pair_corrs = [
        match_mean_abs_corr(a.F_, b.F_) for a, b in itertools.combinations(fits, 2)
    ]
    out["rotation_stability"] = dict(
        mean=round(float(np.mean(pair_corrs)), 3),
        min=round(float(np.min(pair_corrs)), 3),
        pairs=[round(c, 3) for c in pair_corrs],
    )
    print("rotation stability:", out["rotation_stability"])

    # NMF baseline at matched k: fit on log1p normalized, score NB deviance
    # of its implied mean on the count scale using our fitted theta.
    from sklearn.decomposition import NMF

    totals = np.asarray(X.sum(axis=0)).ravel()
    s = totals / np.median(totals)
    Xn = X.multiply(1.0 / s[None, :]).tocsc()
    Y = Xn.copy()
    Y.data = np.log1p(Y.data)
    t0 = time.perf_counter()
    nmf = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=0)
    W = nmf.fit_transform(Y)
    H = nmf.components_
    mu_nmf = np.expm1(np.clip(W @ H, 0, 30)) * s[None, :] + 1e-9
    out["nmf"] = dict(
        wall_s=round(time.perf_counter() - t0, 1),
        dev_expl=round(_dev_expl_of_mu(X, mu_nmf, m.theta_), 4),
    )
    print("nmf:", out["nmf"])

    # GLM-PCA baseline (Poisson/NB GLM factor model, unconstrained loadings).
    # Pure-numpy dense implementation: only run at quick size, where all three
    # methods see identical data; hours-slow beyond ~5e6 entries.
    if p * n > 5_000_000:
        out["glmpca"] = dict(skipped="matrix too large for the dense glmpca reference; see --quick run")
        print("glmpca:", out["glmpca"])
        return out
    try:
        from glmpca.glmpca import glmpca

        t0 = time.perf_counter()
        res = glmpca(np.asarray(X.todense()), L=k, fam="nb", verbose=False)
        mu_g = np.exp(np.clip(_glmpca_eta(res, X), -30, 30))
        out["glmpca"] = dict(
            wall_s=round(time.perf_counter() - t0, 1),
            dev_expl=round(_dev_expl_of_mu(X, mu_g, m.theta_), 4),
        )
    except Exception as e:  # optional dependency; report rather than fail
        out["glmpca"] = dict(error=repr(e)[:200])
    print("glmpca:", out["glmpca"])
    return out


def _glmpca_eta(res, X):
    # glmpca returns factors/loadings; eta = row offsets + loadings @ factors^T
    V = res["loadings"]  # features x k
    U = res["factors"]  # samples x k
    eta = V @ U.T
    totals = np.asarray(X.sum(axis=0)).ravel()
    eta = eta + np.log(totals / np.median(totals))[None, :]
    row_mean = np.log(np.maximum(np.asarray(X.mean(axis=1)).ravel(), 1e-8))
    return eta + row_mean[:, None]


def _dev_expl_of_mu(X, mu, theta) -> float:
    from glm_seminmf._likelihood import nb_deviance

    Xd = torch.as_tensor(np.asarray(X.todense(), dtype=np.float64))
    mu_t = torch.as_tensor(np.asarray(mu, dtype=np.float64))
    th = torch.as_tensor(theta[:, None])
    d_model = nb_deviance(Xd, mu_t, th).sum().item()
    totals = Xd.sum(dim=0)
    null_mu = torch.outer(Xd.sum(dim=1) / totals.sum(), totals) + 1e-9
    d_null = nb_deviance(Xd, null_mu, th).sum().item()
    return 1.0 - d_model / d_null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument(
        "--only", default=None,
        choices=["scaling", "initopt", "exposure", "real"],
    )
    args = ap.parse_args()
    device = args.device

    results = {"device": device, "cuda": torch.cuda.is_available(), "quick": args.quick}
    if args.only in (None, "scaling"):
        results["scaling"] = bench_scaling(args.quick, device)
    if args.only in (None, "initopt"):
        results["init_optimizer"] = bench_init_and_optimizer(args.quick, device)
    if args.only in (None, "exposure"):
        results["exposure"] = bench_exposure(args.quick, device)
    if args.only in (None, "real"):
        results["real"] = bench_real_data(args.quick, device)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
