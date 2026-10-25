"""README figures.   python -m carmatch.report   (after python -m carmatch.price_model)"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from carmatch.data import ROOT, load  # noqa: E402
from carmatch.price_model import NAMES, PriceModel  # noqa: E402

REPORTS, FIGS = ROOT / "reports", ROOT / "reports" / "figures"
GBM, MY, M, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#b8b7b0", "#0b0b0b", "#52514e", "#d9d8d2"
COLS = {"gbm": GBM, "model_year_median": MY, "model_median": M}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def _t(ax, s):
    ax.set_title(s, loc="left", color=INK, fontsize=10)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    m = json.loads((REPORTS / "metrics.json").read_text())
    t = pd.read_csv(REPORTS / "test_predictions.csv.gz")
    order = ["gbm", "model_year_median", "model_median"]

    # 1. accuracy vs baselines
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    for ax, (k, title, fmt) in zip(axes, [("mape", "Average % error (MAPE), held-out adverts", "{:.1f}%"),
                                          ("mae", "Average error in £ (MAE)", "£{:,.0f}")]):
        vals = [m["overall"][o][k] for o in order]
        y = np.arange(3)[::-1]
        ax.barh(y, vals, color=[COLS[o] for o in order], height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v * 1.01, yi, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [NAMES[o] for o in order])
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="y", visible=False)
        _t(ax, title)
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_vs_baselines.png", dpi=140)
    plt.close(fig)

    # 2. predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, sharey=True)
    s = t.sample(min(6000, len(t)), random_state=0)
    for ax, k in zip(axes, ["model_year_median", "gbm"]):
        ax.scatter(s.price / 1e3, s[f"pred_{k}"] / 1e3, s=3, alpha=0.25, color=COLS[k], linewidths=0)
        ax.plot([0, 80], [0, 80], color=INK, lw=0.8)
        ax.set(xlim=(0, 80), ylim=(0, 80), xlabel="asking price (£k)", ylabel="predicted fair price (£k)")
        _t(ax, f"{NAMES[k]}: MAPE {m['overall'][k]['mape']:.1f}%")
    fig.tight_layout()
    fig.savefig(FIGS / "predicted_vs_actual.png", dpi=140)
    plt.close(fig)

    # 3. error by price band and brand
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    pb = pd.DataFrame(m["by_price_band"]).T
    x = np.arange(len(pb))
    for i, k in enumerate(["model_year_median", "gbm"]):
        axes[0].bar(x + (i - 0.5) * 0.38, pb[k], width=0.38, color=COLS[k], label=NAMES[k])
    axes[0].set_xticks(x, [f"{b}\n(n={int(n):,})" for b, n in zip(pb.index, pb.n)])
    axes[0].set_ylabel("MAPE (%)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="x", visible=False)
    _t(axes[0], "Error by price band")
    br = pd.DataFrame(m["by_brand"]).T.sort_values("gbm")
    y = np.arange(len(br))
    axes[1].barh(y + 0.2, br.model_year_median, height=0.38, color=COLS["model_year_median"], label=NAMES["model_year_median"])
    axes[1].barh(y - 0.2, br.gbm, height=0.38, color=GBM, label=NAMES["gbm"])
    axes[1].set_yticks(y, br.index)
    axes[1].set_xlabel("MAPE (%)")
    axes[1].grid(axis="y", visible=False)
    _t(axes[1], "Error by brand")
    fig.tight_layout()
    fig.savefig(FIGS / "error_by_band_and_brand.png", dpi=140)
    plt.close(fig)

    # 4. depreciation curves from the model (typical spec, 8,000 miles a year)
    df, _ = load()
    pm = PriceModel.load()
    pick = ["Ford Fiesta", "Volkswagen Golf", "Toyota Yaris", "BMW 3 Series", "Skoda Octavia", "Mercedes-Benz C Class"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6", "#d6a12a", "#52514e"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = []
    for mk, col in zip(pick, cols):
        sub = df[df.model_key == mk]
        if sub.empty:
            continue
        typ = sub.mode().iloc[0]
        ages = np.arange(0, 11)
        f = pd.DataFrame([{**typ.to_dict(), "age": a, "mileage": max(500, a * 8000)} for a in ages])
        f["engine_size"] = sub.engine_size.median()
        f["mpg"] = sub.mpg.median()
        p = pm.predict(f)
        ax.plot(ages, p / p[0] * 100, color=col, lw=2, label=f"{mk} ({typ.fuel.lower()}, {typ.transmission.lower()})")
        rows.append({"model": mk, "value_after_3y_pct": float(p[3] / p[0] * 100), "value_after_6y_pct": float(p[6] / p[0] * 100)})
    ax.set(xlabel="age (years), at 8,000 miles a year", ylabel="% of value when nearly new", ylim=(0, 105))
    ax.legend(frameon=False, fontsize=7.5)
    _t(ax, "How fast they lose value: the price model's depreciation curves")
    fig.tight_layout()
    fig.savefig(FIGS / "depreciation_curves.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORTS / "depreciation_examples.csv", index=False)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
