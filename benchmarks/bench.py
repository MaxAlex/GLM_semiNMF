"""Benchmarks (spec section 7): timing/memory scaling, initialization
comparison, offset vs fitted exposure, deviance vs NMF / GLM-PCA, rotation
stability on real data.

Run:  uv run --group benchmark python benchmarks/bench.py [--quick] [--out DIR]

Solver correctness, stationarity and the penalty grid live in
``bench_optimizer.py``. There is only one solver now: the legacy
``algorithm``/``g_parametrization`` names are deprecated aliases of it, so
sweeping them here would compare a configuration against itself.

Every fit below reports ``stop_reason_`` and the physical residual alongside
wall time. A budget exit is an unresolved run, not a result.

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
try:
    from .scoring import fit_common_null, predictor, score_eta
except ImportError:  # direct script execution
    from scoring import fit_common_null, predictor, score_eta

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


def bench_scaling(quick: bool, device: str, budget: dict):
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
        m = NBGLMSemiNMF(n_components=k, l1_F=0.001 * n, random_state=0, device=device, **budget)
        wall, mem = _fit_timed(m, sim.X)
        rows.append(
            dict(p=p, n=n, k=k, device=device, wall_s=round(wall, 1),
                 peak_gpu_mb=round(mem, 1), n_iter=m.n_iter_,
                 converged=m.converged_, stop_reason=m.stop_reason_,
                 max_residual=float(m.stationarity_["max_residual"]),
                 dtype=m.compute_dtype_, phase_seconds=m.phase_seconds_,
                 total_seconds=round(m.total_seconds_, 1),
                 dev_expl=round(m.deviance_explained_, 3))
        )
        print("scaling:", rows[-1])
    return rows


def bench_init(quick: bool, device: str, budget: dict):
    """Convergence trace + recovery per initialization (spec section 3).

    Compared at equal stationarity: a start that stops at max_iter with a
    large residual has not been shown to be a worse optimum, only a slower
    one, so the residual is reported next to the loss.
    """
    p, n, k = (600, 2000, 6) if quick else (2000, 8000, 8)
    sim = simulate_nb_seminmf(p=p, n=n, k=k, random_state=1)
    lam = 0.001 * n
    out = {"traces": {}, "summary": []}

    for init in ["svd", "nmf", "random"]:
        m = NBGLMSemiNMF(n_components=k, l1_F=lam, init=init,
                         random_state=0, device=device, **budget)
        wall, _ = _fit_timed(m, sim.X)
        rec = match_mean_abs_corr(sim.F, m.F_)
        out["traces"][init] = list(m.loss_)
        out["summary"].append(
            dict(init=init, wall_s=round(wall, 1), n_iter=m.n_iter_,
                 converged=m.converged_, stop_reason=m.stop_reason_,
                 max_residual=float(m.stationarity_["max_residual"]),
                 safeguard_active=bool(m.stationarity_["safeguard_active"]),
                 initial_objective=round(m.initial_objective_, 1),
                 final_objective=round(m.final_objective_, 1),
                 recovery=round(rec, 3),
                 G_exact_zero_frac=round(float((m.G_ == 0).mean()), 3))
        )
        print("init:", out["summary"][-1])
    return out


def bench_exposure(quick: bool, device: str, budget: dict):
    """Open question 5: does fitting b change recovered factors materially?"""
    p, n, k = (600, 2000, 6) if quick else (2000, 8000, 8)
    sim = simulate_nb_seminmf(p=p, n=n, k=k, random_state=2, exposure_sd=0.8)
    fits, walls = {}, {}
    for mode in ["offset", "fit"]:
        m = NBGLMSemiNMF(n_components=k, l1_F=0.001 * n, exposure=mode,
                         random_state=0, device=device, **budget)
        wall, _ = _fit_timed(m, sim.X)
        fits[mode] = m
        walls[mode] = round(wall, 1)
        print(f"exposure={mode}: {wall:.1f}s recovery="
              f"{match_mean_abs_corr(sim.F, m.F_):.3f} dev_expl={m.deviance_explained_:.3f} "
              f"stop={m.stop_reason_} residual={m.stationarity_['max_residual']:.3g}")
    cross = match_mean_abs_corr(fits["offset"].F_, fits["fit"].F_)
    b_corr = float(np.corrcoef(fits["offset"].b_, fits["fit"].b_)[0, 1])
    return dict(
        recovery_offset=round(match_mean_abs_corr(sim.F, fits["offset"].F_), 3),
        recovery_fit=round(match_mean_abs_corr(sim.F, fits["fit"].F_), 3),
        cross_agreement=round(cross, 3), b_agreement=round(b_corr, 3),
        wall_s={k: v for k, v in walls.items()},
        # Equal-stationarity caveat: wall times are comparable only when both
        # modes reached the same residual.
        stop_reason={k: v.stop_reason_ for k, v in fits.items()},
        max_residual={k: float(v.stationarity_["max_residual"]) for k, v in fits.items()},
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


def bench_real_data(quick: bool, device: str, budget: dict, glmpca_max_iter: int = 300):
    X = load_newsgroups(quick)
    p, n = X.shape
    k = 10
    lam = 0.001 * n
    print(f"20 Newsgroups: p={p} n={n} nnz_frac={X.nnz/(p*n):.4f}")
    out = {"shape": [p, n]}
    totals = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)
    b_common = np.log(totals / np.median(totals))
    common = fit_common_null(X, b_common, device=device,
                             max_seconds=budget.get('max_seconds'),
                             max_iter=budget.get('max_iter', 500))
    common_eta = predictor(common, b_common)
    out['common_null'] = dict(stop_reason=common.stop_reason_, converged=common.converged_,
                              dispersion='shared', theta=common.theta_.tolist())
    common_score = lambda eta: score_eta(X, eta, common_eta, common.theta_)[
        'common_theta_deviance_explained']

    m = NBGLMSemiNMF(n_components=k, l1_F=lam, random_state=0, device=device, **budget)
    wall, mem = _fit_timed(m, X)
    dev_ours = m.deviance_explained_
    out["ours"] = dict(wall_s=round(wall, 1), dev_expl=round(dev_ours, 4),
                       n_iter=m.n_iter_, peak_gpu_mb=round(mem, 1),
                       converged=m.converged_, stop_reason=m.stop_reason_,
                       max_residual=float(m.stationarity_["max_residual"]),
                       safeguard_active=bool(m.stationarity_["safeguard_active"]),
                       total_seconds=round(m.total_seconds_, 1))
    out['ours']['common_theta_dev_expl'] = common_score(predictor(m, m.b_, m.G_))
    print("ours:", out["ours"])

    # Rotation stability needs genuinely distinct starts. random_state alone
    # does not give them: with init="svd" the seed only sets the starting
    # vector for `svds`, which converges to the same subspace, so starts
    # differ by ~1e-14 relative and match at |corr| = 1.000000. Agreement
    # between such runs measures determinism, not identification. Perturb the
    # warm start per replicate instead, as bench_optimizer.py does.
    from glm_seminmf._init import initialize

    totals = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1.0)
    b0 = np.log(totals / np.median(totals))
    F0, G0, _, _ = initialize(X, b0, k, "svd", 0)
    seeds = range(3 if quick else 5)
    fits, start_agreement = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        Fs = F0 + rng.normal(0.0, 0.05 * np.abs(F0).mean(), F0.shape)
        Gs = np.maximum(G0 + rng.normal(0.0, 0.05 * G0.mean(), G0.shape), 0.0)
        start_agreement.append(match_mean_abs_corr(F0, Fs))
        ms = NBGLMSemiNMF(n_components=k, l1_F=lam, init=(Fs, Gs),
                          random_state=s, device=device, **budget)
        ms.fit(X)
        fits.append(ms)
    pair_corrs = [
        match_mean_abs_corr(a.F_, b.F_) for a, b in itertools.combinations(fits, 2)
    ]
    out["rotation_stability"] = dict(
        mean=round(float(np.mean(pair_corrs)), 3),
        min=round(float(np.min(pair_corrs)), 3),
        pairs=[round(c, 3) for c in pair_corrs],
        # Agreement between restarts that all stopped short of stationarity is
        # not evidence of identification: repeated non-convergence from
        # near-identical starts looks the same as a recovered optimum.
        stop_reason=[f.stop_reason_ for f in fits],
        max_residual=[float(f.stationarity_["max_residual"]) for f in fits],
        all_certified=all(f.converged_ for f in fits),
        # How far apart the starts actually were, so the result can be read as
        # "agreement despite this much perturbation" rather than as a bare 1.0.
        start_agreement_to_unperturbed=[round(c, 4) for c in start_agreement],
    )
    print("rotation stability:", out["rotation_stability"])

    # NMF baseline at matched k: fit on log1p normalized, score NB deviance
    # of its implied mean at the independent null's common theta.
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
        common_theta_dev_expl=common_score(np.log(mu_nmf)),
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
        # Bound the reference the same way ours is bounded, and record the
        # bound: an unbounded baseline against a budgeted fit is not a
        # comparison. glmpca has no stationarity report of its own, so its
        # iteration cap is the only budget statement available for it.
        ctl = {"maxIter": glmpca_max_iter, "eps": 1e-4}
        np.random.seed(0)
        res = glmpca(np.asarray(X.todense()), L=k, fam="nb", ctl=ctl, verbose=False,
                     sz=np.exp(b_common))
        eta_g = _glmpca_eta(res, X, size_factors=np.exp(b_common))
        out["glmpca"] = dict(
            wall_s=round(time.perf_counter() - t0, 1),
            max_iter=glmpca_max_iter,
            common_theta_dev_expl=common_score(eta_g),
        )
    except Exception as e:  # optional dependency; report rather than fail
        out["glmpca"] = dict(error=repr(e)[:200])
    print("glmpca:", out["glmpca"])
    return out


def _glmpca_eta(res, X, *, size_factors=None):
    # Verified against glmpca-py 0.1.0 glmpca_init/postprocess: for the no-
    # covariate benchmark, eta = log(sz) + fitted intercept + V @ U.T.
    # Its default sz is column MEANS, not median-normalized library totals.
    # https://github.com/willtownes/glmpca-py/blob/master/glmpca/glmpca.py
    V = res["loadings"]  # features x k
    U = res["factors"]  # samples x k
    intercept = np.asarray(res['coefX'])
    if intercept.shape != (X.shape[0], 1) or res.get('coefZ') is not None:
        raise ValueError('This reconstruction supports the intercept-only GLM-PCA benchmark')
    sz = np.asarray(X.mean(axis=0)).ravel() if size_factors is None else np.asarray(size_factors)
    if sz.shape != (X.shape[1],) or (sz <= 0).any() or not np.isfinite(sz).all():
        raise ValueError('GLM-PCA size factors must be finite and positive')
    return V @ U.T + intercept + np.log(sz)[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    # An unstated budget makes every comparison below a comparison of
    # transients. Both are recorded in the output for provenance.
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--glmpca-max-iter", type=int, default=300)
    # The "auto" usage penalty (0.005*p) is calibrated on the synthetic
    # generator's count scale. On much sparser data it can drive every usage to
    # zero, which certifies as a degenerate optimum, so it must be settable.
    ap.add_argument("--l1-G", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument(
        "--only", default=None,
        choices=["scaling", "init", "exposure", "real"],
    )
    args = ap.parse_args()
    device = args.device

    budget = {"max_iter": args.max_iter, "max_seconds": args.max_seconds}
    if args.l1_G is not None:
        budget["l1_G"] = args.l1_G
    results = {"device": device, "cuda": torch.cuda.is_available(), "quick": args.quick,
               "budget": budget}
    if args.only in (None, "scaling"):
        results["scaling"] = bench_scaling(args.quick, device, budget)
    if args.only in (None, "init"):
        results["init"] = bench_init(args.quick, device, budget)
    if args.only in (None, "exposure"):
        results["exposure"] = bench_exposure(args.quick, device, budget)
    if args.only in (None, "real"):
        results["real"] = bench_real_data(args.quick, device, budget, args.glmpca_max_iter)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
