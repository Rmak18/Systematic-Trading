
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_YIELD_CSV = os.path.join("data", "yield_curve.csv")


def _ns_loadings(maturity_years: np.ndarray, lam: float = 0.0609) -> np.ndarray:
    tau = np.asarray(maturity_years, dtype=float)
    denom = np.where(tau * lam == 0, 1e-12, tau * lam)
    exp_term = np.exp(-lam * tau)
    L1 = np.ones_like(tau)
    L2 = (1 - exp_term) / denom
    L3 = L2 - exp_term
    X = np.column_stack([L1, L2, L3])
    return X


def _parse_maturity_label(label: str) -> Optional[float]:
    """Best-effort maturity parser accepting names like '10Y', 'yield_2y', '3m'."""
    if not isinstance(label, str):
        return None
    cleaned = label.strip().lower()
    for prefix in ["yield_", "yc_", "rate_", "yield", "yc", "rate"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    cleaned = cleaned.replace("-", "").replace("year", "y")
    if cleaned.endswith("y"):
        num = cleaned[:-1]
        try:
            return float(num)
        except ValueError:
            return None
    if cleaned.endswith("yr"):
        num = cleaned[:-2]
        try:
            return float(num)
        except ValueError:
            return None
    if cleaned.endswith("m"):
        num = cleaned[:-1]
        try:
            return float(num) / 12.0
        except ValueError:
            return None
    return None


def estimate_ns_factors(yields_df: pd.DataFrame, lam: float = 0.0609) -> pd.DataFrame:
    maturity_map = {
        "1M": 1 / 12, "3M": 3 / 12, "6M": 6 / 12,
        "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0,
        "10Y": 10.0, "20Y": 20.0, "30Y": 30.0
    }
    maturity_pairs: List[Tuple[str, float]] = []
    for col in yields_df.columns:
        tau = maturity_map.get(col)
        if tau is None:
            tau = _parse_maturity_label(col)
        if tau is not None and np.isfinite(tau):
            maturity_pairs.append((col, float(tau)))

    if not maturity_pairs:
        raise ValueError(
            f"No recognizable maturity columns found. Columns were: {list(yields_df.columns)}"
        )

    maturity_pairs.sort(key=lambda x: x[1])
    cols = [c for c, _ in maturity_pairs]

    if len(cols) < 3:
        # Fallback: derive simple proxies so downstream logic still works.
        longest = maturity_pairs[-1][0]
        out = pd.DataFrame(index=yields_df.index)
        out["NS_level"] = yields_df[longest].astype(float)
        if len(cols) >= 2:
            shortest = maturity_pairs[0][0]
            out["NS_slope"] = (
                yields_df[longest].astype(float) - yields_df[shortest].astype(float)
            )
        else:
            out["NS_slope"] = 0.0
        out["NS_curvature"] = 0.0
        return out

    taus = np.array([tau for _, tau in maturity_pairs], dtype=float)
    X = _ns_loadings(taus, lam=lam)

    betas = []
    subset = yields_df[cols].astype(float)
    for _, row in subset.iterrows():
        y = row.values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas.append(b)

    betas = np.asarray(betas)
    out = pd.DataFrame(
        betas, columns=["NS_level", "NS_slope", "NS_curvature"], index=yields_df.index
    )
    return out


def fit_ar1_nll(series: pd.Series) -> Tuple[float, Dict[str, float]]:
    x = series.dropna().values.astype(float)
    if len(x) < 30:
        return 1e9, {"a": 0.0, "phi": 0.0, "sigma2": 1.0, "n": 0}

    x_t = x[1:]
    x_lag = x[:-1]
    X = np.column_stack([np.ones_like(x_lag), x_lag])
    b, *_ = np.linalg.lstsq(X, x_t, rcond=None)
    a, phi = b
    resid = x_t - (a + phi * x_lag)
    n = len(resid)
    sigma2 = np.mean(resid ** 2)
    if sigma2 <= 0 or not np.isfinite(sigma2):
        sigma2 = 1e-6
    nll = 0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    return nll, {"a": float(a), "phi": float(phi), "sigma2": float(sigma2), "n": int(n)}


def nll_for_factors(factors_df: pd.DataFrame, idx: np.ndarray) -> float:
    if idx.size < 40:
        return 1e9
    sub = factors_df.iloc[idx]
    total = 0.0
    for col in ["NS_level", "NS_slope", "NS_curvature"]:
        nll, _ = fit_ar1_nll(sub[col])
        total += nll
    return float(total)


from dataclasses import dataclass

@dataclass
class Split:
    feature: str
    threshold: float


@dataclass
class TreeNode:
    idx: np.ndarray
    depth: int
    split: Optional[Split] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    nll: Optional[float] = None


class StateGateTree:
    def __init__(self, max_leaves: int = 3, min_rel_improve: float = 0.02,
                 candidate_quantiles: List[float] = [0.2, 0.4, 0.6, 0.8]):
        self.max_leaves = max_leaves
        self.min_rel_improve = min_rel_improve
        self.candidate_quantiles = candidate_quantiles
        self.root: Optional[TreeNode] = None
        self.features_: List[str] = []

    def fit(self, factors_df: pd.DataFrame, features_df: pd.DataFrame):
        assert (factors_df.index == features_df.index).all(), "Indices must match"

        n = len(factors_df)
        all_idx = np.arange(n, dtype=int)

        self.features_ = list(features_df.columns)

        root = TreeNode(idx=all_idx, depth=0, split=None, left=None, right=None)
        root.nll = nll_for_factors(factors_df, root.idx)

        leaves: List[TreeNode] = [root]

        while len(leaves) < self.max_leaves:
            best = None

            for leaf in leaves:
                if leaf.idx.size < 80:
                    continue
                cand = self._best_split_for_node(leaf, factors_df, features_df)
                if cand is None:
                    continue
                improve, feat, thr, left_idx, right_idx, left_nll, right_nll = cand
                if best is None or improve > best[0]:
                    best = cand + (leaf,)

            if best is None:
                break

            improve, feat, thr, left_idx, right_idx, left_nll, right_nll, leaf_to_split = best
            rel_improve = improve / max(leaf_to_split.nll, 1e-9)
            if rel_improve < self.min_rel_improve:
                break

            leaf_to_split.split = Split(feat, float(thr))
            left = TreeNode(idx=left_idx, depth=leaf_to_split.depth + 1, nll=left_nll)
            right = TreeNode(idx=right_idx, depth=leaf_to_split.depth + 1, nll=right_nll)
            leaf_to_split.left = left
            leaf_to_split.right = right

            leaves.remove(leaf_to_split)
            leaves.extend([left, right])

        self.root = root

    def _best_split_for_node(self, node: 'TreeNode',
                             factors_df: pd.DataFrame,
                             features_df: pd.DataFrame):
        parent_nll = node.nll if node.nll is not None else nll_for_factors(factors_df, node.idx)
        best = None
        sub_feats = features_df.iloc[node.idx]

        for feat in self.features_:
            x = sub_feats[feat].values.astype(float)
            mask_ok = np.isfinite(x)
            if mask_ok.sum() < 80:
                continue
            x_ok = x[mask_ok]
            idx_ok = node.idx[mask_ok]

            try:
                qs = np.unique(np.quantile(x_ok, [0.2, 0.4, 0.6, 0.8]))
            except Exception:
                continue
            for thr in qs:
                left_mask = x_ok <= thr
                right_mask = x_ok > thr
                if left_mask.sum() < 40 or right_mask.sum() < 40:
                    continue
                left_idx = idx_ok[left_mask]
                right_idx = idx_ok[right_mask]
                left_nll = nll_for_factors(factors_df, left_idx)
                right_nll = nll_for_factors(factors_df, right_idx)
                child_nll = left_nll + right_nll
                improve = parent_nll - child_nll
                if best is None or improve > best[0]:
                    best = (improve, feat, thr, left_idx, right_idx, left_nll, right_nll)

        return best

    def predict_regime_indices(self, features_df: pd.DataFrame) -> np.ndarray:
        leaves = []
        def visit(node):
            if node is None:
                return
            if node.split is None:
                leaves.append(node)
            else:
                visit(node.left)
                visit(node.right)
        visit(self.root)

        leaf_to_id = {id(leaf): i for i, leaf in enumerate(leaves)}

        regimes = []
        n = len(features_df)
        node = self.root
        for i in range(n):
            node = self.root
            while node.split is not None:
                feat = node.split.feature
                thr = node.split.threshold
                val = features_df.iloc[i][feat]
                go_left = (val <= thr) if np.isfinite(val) else True
                node = node.left if go_left else node.right
            regimes.append(leaf_to_id[id(node)])
        return np.array(regimes, dtype=int)

    def to_dict(self):
        def encode(node):
            if node is None:
                return None
            if node.split is None:
                return {"leaf": True, "n": int(node.idx.size), "nll": float(node.nll)}
            return {
                "leaf": False,
                "feature": node.split.feature,
                "threshold": float(node.split.threshold),
                "nll": float(node.nll) if node.nll is not None else None,
                "left": encode(node.left),
                "right": encode(node.right),
            }
        return encode(self.root)


def build_features(yields_df: pd.DataFrame, macro_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if macro_df is not None:
        df = yields_df.join(macro_df, how="inner")
        num = df.select_dtypes(include=[np.number])
        num.index = df.index
        return num
    else:
        df = pd.DataFrame(index=yields_df.index)
        if "10Y" in yields_df.columns:
            df["Y10"] = yields_df["10Y"]
        if set(["10Y", "2Y"]).issubset(yields_df.columns):
            df["Slope_10y_2y"] = yields_df["10Y"] - yields_df["2Y"]
        if set(["5Y", "2Y", "10Y"]).issubset(yields_df.columns):
            df["Curvature_proxy"] = 2 * yields_df["5Y"] - yields_df["2Y"] - yields_df["10Y"]
        if df.shape[1] == 0:
            first_col = yields_df.columns[0]
            df["Level_proxy"] = yields_df[first_col]
        return df.dropna(how="all")
    
def compute_tsmom_scale(yields_df: pd.DataFrame,
                         prefer_col: str = "10Y",
                         lookback: int = 12,
                         vol_lookback: int = 24) -> float:
  
    if yields_df is None or yields_df.empty:
        return 0.5

    col = None
    if prefer_col in yields_df.columns:
        col = prefer_col
    else:
        def _tenor_years(name: str) -> float:
            name = str(name).strip().upper()
            if name.endswith("Y"):
                try:
                    return float(name[:-1])
                except Exception:
                    return float("nan")
            if name.endswith("M"):
                try:
                    return float(name[:-1]) / 12.0
                except Exception:
                    return float("nan")
            return float("nan")
     
        tenor_pairs = [(_tenor_years(c), c) for c in yields_df.columns]
        tenor_pairs = [(t, c) for t, c in tenor_pairs if np.isfinite(t)]
        if tenor_pairs:
            col = max(tenor_pairs, key=lambda x: x[0])[1]

    if col is None:
   
        num_cols = yields_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return 0.5
        col = num_cols[0]

    s = pd.Series(yields_df[col]).dropna()
    if s.size < max(lookback + 1, vol_lookback + 2):
        return 0.5

   
    mom = -(s.iloc[-1] - s.iloc[-lookback])

   
    vol = s.diff().rolling(vol_lookback).std().iloc[-1]
    if not np.isfinite(vol) or vol <= 1e-12:
        vol = s.diff().std()
    if not np.isfinite(vol) or vol <= 1e-12:
        return 0.5

    z = abs(mom) / (vol * math.sqrt(max(1, lookback)))
    
    scale = math.tanh(0.5 * z)
    scale = float(np.clip(scale, 0.0, 1.0))
    return scale



def learn_state_gate(yields_df: pd.DataFrame,
                     macro_df: Optional[pd.DataFrame] = None,
                     lam: float = 0.0609,
                     max_leaves: int = 3,
                     min_rel_improve: float = 0.02):
    factors = estimate_ns_factors(yields_df, lam=lam)
    features = build_features(yields_df, macro_df)
    df_all = factors.join(features, how="inner")
    factors = df_all[["NS_level", "NS_slope", "NS_curvature"]]
    features = df_all.drop(columns=["NS_level", "NS_slope", "NS_curvature"], errors="ignore")
    tree = StateGateTree(max_leaves=max_leaves, min_rel_improve=min_rel_improve)
    tree.fit(factors, features)
    regimes = tree.predict_regime_indices(features)
    out = pd.DataFrame({"Regime": regimes}, index=factors.index)
    
    return factors, features, tree, out


def plot_regimes(yields_df: pd.DataFrame, regimes_df: pd.DataFrame, out_png: Optional[str] = None):
    df = yields_df.join(regimes_df, how="inner")
    if "10Y" in df.columns:
        y = df["10Y"]
        y_label = "10Y Yield (%)"
    else:
        y = None
        y_label = "NS Level (approx %)"
    r = df["Regime"].values
    dates = df.index

    plt.figure(figsize=(11, 5))
    if y is not None:
        plt.plot(dates, y, linewidth=1.5)
    else:
        plt.plot(dates, np.zeros_like(r), linewidth=1.5)

    unique_regs = np.unique(r)
    for reg in unique_regs:
        mask = (r == reg)
        start = None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            if (not m or i == len(mask) - 1) and start is not None:
                end = i if not m else i
                plt.axvspan(dates[start], dates[end], alpha=0.1)
                start = None

    plt.title("US Treasury Yield & State Gate Regimes")
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    return plt.gcf()


def _read_csv_maybe(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or path == "":
        return None
    df = pd.read_csv(path)
    date_col = None
    for cand in ["Date", "date", "DATE", "dt", "time", "Time", "TIME"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"{path}: expected a 'Date' column")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


def run_demo(yields_csv: str = DEFAULT_YIELD_CSV,
             macro_csv: Optional[str] = None,
             out_regimes_csv: str = "regimes.csv",
             out_tree_json: str = "state_gate_tree.json",
             out_plot_png: Optional[str] = "state_gate_plot.png",
             lam: float = 0.0609,
             max_leaves: int = 3,
             min_rel_improve: float = 0.02):
    yields_df = _read_csv_maybe(yields_csv)
    macro_df = _read_csv_maybe(macro_csv) if macro_csv else None

    factors, features, tree, regimes = learn_state_gate(
        yields_df, macro_df, lam=lam, max_leaves=max_leaves, min_rel_improve=min_rel_improve
    )

    regimes_out = regimes.copy()
    regimes_out.index.name = "Date"
    regimes_out.to_csv(out_regimes_csv)

    with open(out_tree_json, "w") as f:
        json.dump(tree.to_dict(), f, indent=2)

    try:
        plot_regimes(yields_df, regimes_out, out_plot_png)
    except Exception as e:
        print("Plotting skipped:", e)
        
    try:
        scale = compute_tsmom_scale(yields_df)
        print(f"TSMOM_scale={scale:.4f}")
    except Exception as e:
        print("TSMOM scale not available:", e)
   


    return factors, features, tree, regimes_out


def build_templates():
    dates = pd.date_range("2015-01-01", periods=24, freq="M")
    rng = np.random.default_rng(0)
    base_level = 2.0 + 0.4 * np.sin(np.linspace(0, 6.0, len(dates)))
    ten_y = base_level + rng.normal(0, 0.1, len(dates))
    two_y = base_level - 0.6 + rng.normal(0, 0.1, len(dates))
    three_m = base_level - 1.2 + rng.normal(0, 0.1, len(dates))

    yields_t = pd.DataFrame({
        "Date": dates,
        "3M": three_m,
        "2Y": two_y,
        "10Y": ten_y,
    })
    yields_t.to_csv("yields_template.csv", index=False)

    macro_t = pd.DataFrame({
        "Date": dates,
        "CPI_YoY": 2.0 + 0.3 * np.sin(np.linspace(0, 4.0, len(dates))) + rng.normal(0, 0.1, len(dates)),
        "Unemployment": 5.5 + 0.5 * np.cos(np.linspace(0, 3.0, len(dates))) + rng.normal(0, 0.1, len(dates)),
        "FedFunds": 1.0 + 0.5 * np.sin(np.linspace(0, 5.0, len(dates))) + rng.normal(0, 0.1, len(dates)),
    })
    macro_t.to_csv("macro_template.csv", index=False)


def cli():
    parser = argparse.ArgumentParser(description="State gate for US Treasury yields (beginner-friendly).")
    parser.add_argument(
        "--yields_csv",
        default=DEFAULT_YIELD_CSV,
        help=f"CSV with Date and maturities (default: {DEFAULT_YIELD_CSV})."
    )
    parser.add_argument("--macro_csv", default=None, help="CSV with Date and macro features (optional).")
    parser.add_argument("--out_regimes_csv", default="regimes.csv", help="Where to save regimes timeline.")
    parser.add_argument("--out_tree_json", default="state_gate_tree.json", help="Where to save the learned tree.")
    parser.add_argument("--plot_png", default="state_gate_plot.png", help="Where to save the plot (PNG).")
    parser.add_argument("--lam", type=float, default=0.0609, help="NS decay parameter lambda.")
    parser.add_argument("--max_leaves", type=int, default=3, help="Max number of regimes (leaves).")
    parser.add_argument("--min_rel_improve", type=float, default=0.02, help="Minimum relative NLL improvement to split.")
    parser.add_argument("--build_templates", action="store_true", help="Write example CSV templates to current folder.")
    args = parser.parse_args()

    if args.build_templates:
        build_templates()
        print("Templates written: yields_template.csv, macro_template.csv")

    run_demo(
        yields_csv=args.yields_csv,
        macro_csv=args.macro_csv,
        out_regimes_csv=args.out_regimes_csv,
        out_tree_json=args.out_tree_json,
        out_plot_png=args.plot_png,
        lam=args.lam,
        max_leaves=args.max_leaves,
        min_rel_improve=args.min_rel_improve,
    )


if __name__ == "__main__":
    cli()
