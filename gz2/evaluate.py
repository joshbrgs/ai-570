"""Framework-neutral evaluation from saved predictions: metrics, breakdowns and paired comparison.

A predictions frame has one row per test galaxy with columns
`dr7objid, bar_true, bar_w, bar_prob, winding_w, t10_*_true, t10_*_pred`
(written by `gz2.jax_train.predict`). Any model that writes this format can be compared.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gz2.data import ROOT, WINDING_COLS, load_galaxies, load_photometry

PREDICTIONS_DIR = ROOT / "output" / "predictions"
CONFIDENT = 0.8
TRUE_COLS = [f"{c}_true" for c in WINDING_COLS]
PRED_COLS = [f"{c}_pred" for c in WINDING_COLS]


def metrics(preds: pd.DataFrame, loss_weights: tuple[float, float] = (1.0, 0.5)) -> dict:
    """Masked metrics: each head is scored only on galaxies that have its label (weight > 0)."""
    bar = preds[preds["bar_w"] > 0]
    winding = preds[preds["winding_w"] > 0]
    out = {"n_bar": len(bar), "n_winding": len(winding)}

    if len(bar):
        p = bar["bar_prob"].clip(1e-7, 1 - 1e-7).to_numpy()
        y = bar["bar_true"].to_numpy()
        out["bar_loss"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out["bar_acc"] = float(((p >= 0.5) == (y == 1)).mean())
        out["bar_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    else:
        out.update(bar_loss=np.nan, bar_acc=np.nan, bar_auc=np.nan)

    if len(winding):
        err = winding[PRED_COLS].to_numpy() - winding[TRUE_COLS].to_numpy()
        out["winding_loss"] = float((err**2).mean())
        out["winding_rmse"] = float(np.sqrt((err**2).mean()))
        out["winding_mae"] = float(np.abs(err).mean())
    else:
        out.update(winding_loss=np.nan, winding_rmse=np.nan, winding_mae=np.nan)

    out["loss"] = loss_weights[0] * np.nan_to_num(out["bar_loss"]) + loss_weights[1] * np.nan_to_num(out["winding_loss"])
    return out


def load_context() -> pd.DataFrame:
    """Per-galaxy info for breakdowns: raw debiased bar fraction and native resolution."""
    galaxies = load_galaxies()[["dr7objid", "t03_bar"]]
    return galaxies.merge(load_photometry()[["dr7objid", "native_px"]], on="dr7objid", how="left")


def breakdown(preds: pd.DataFrame, context: pd.DataFrame, n_size_bins: int = 5) -> pd.DataFrame:
    """Metrics overall, on confident labels only, and per galaxy-size quintile.

    The EDA showed labels skew toward large, well-resolved galaxies, so an overall number can hide
    where one architecture wins or loses.
    """
    df = preds.merge(context, on="dr7objid", how="left", validate="one_to_one")
    rows = {"all": metrics(df)}

    confident = df.copy()
    confident["bar_w"] *= (df["t03_bar"] >= CONFIDENT) | (df["t03_bar"] <= 1 - CONFIDENT)
    confident["winding_w"] *= df[TRUE_COLS].max(axis=1) >= CONFIDENT
    rows[f"confident (≥ {CONFIDENT})"] = metrics(confident)

    for interval, part in df.groupby(pd.qcut(df["native_px"], n_size_bins), observed=True):
        rows[f"size {interval.left:.0f}–{interval.right:.0f} px"] = metrics(part)

    return pd.DataFrame(rows).T[["n_bar", "bar_acc", "bar_auc", "n_winding", "winding_rmse", "winding_mae"]]


def save_predictions(preds: pd.DataFrame, model_name: str, seed: int) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{model_name}_seed{seed}.csv"
    preds.to_csv(path, index=False)
    return path


def load_predictions(model_name: str) -> list[pd.DataFrame]:
    """All saved seeds for one model, e.g. `load_predictions("jax_cnn")`."""
    return [pd.read_csv(p) for p in sorted(PREDICTIONS_DIR.glob(f"{model_name}_seed*.csv"))]


def per_galaxy_errors(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-galaxy bar correctness and winding squared error, averaged over seeds."""
    frames = []
    for p in runs:
        frames.append(pd.DataFrame({
            "dr7objid": p["dr7objid"],
            "bar_w": p["bar_w"],
            "winding_w": p["winding_w"],
            "bar_correct": ((p["bar_prob"] >= 0.5) == (p["bar_true"] == 1)).astype(float),
            "winding_sq_err": ((p[PRED_COLS].to_numpy() - p[TRUE_COLS].to_numpy()) ** 2).mean(axis=1),
        }))
    return pd.concat(frames).groupby("dr7objid").mean()


def paired_bootstrap(
    runs_a: list[pd.DataFrame], runs_b: list[pd.DataFrame], n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Difference B − A on the same test galaxies, with a 95% bootstrap confidence interval.

    Galaxies are resampled jointly for both models (a paired test), so the interval reflects
    test-set sampling noise; seed-to-seed variation is averaged into each galaxy's error first.
    For bar accuracy a positive difference favours B; for winding RMSE a negative one does.
    If the interval excludes 0, the difference is unlikely to be test-set noise.
    """
    a = per_galaxy_errors(runs_a)
    b = per_galaxy_errors(runs_b).loc[a.index]
    rng = np.random.default_rng(seed)

    rows = {}
    for name, col, weight_col, transform in [
        ("bar accuracy", "bar_correct", "bar_w", lambda m: m),
        ("winding RMSE", "winding_sq_err", "winding_w", np.sqrt),
    ]:
        mask = a[weight_col] > 0
        xa, xb = a.loc[mask, col].to_numpy(), b.loc[mask, col].to_numpy()
        idx = rng.integers(0, len(xa), size=(n_boot, len(xa)))
        diffs = transform(xb[idx].mean(axis=1)) - transform(xa[idx].mean(axis=1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        rows[name] = {
            "A": transform(xa.mean()),
            "B": transform(xb.mean()),
            "B − A": transform(xb.mean()) - transform(xa.mean()),
            "95% CI low": low,
            "95% CI high": high,
            "n galaxies": int(mask.sum()),
        }
    return pd.DataFrame(rows).T
