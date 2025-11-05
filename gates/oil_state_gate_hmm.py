import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


@dataclass
class HMMParameters:
    n_states: int = 2
    transition_matrix: Optional[np.ndarray] = None
    means: Optional[np.ndarray] = None
    stds: Optional[np.ndarray] = None
    initial_probs: Optional[np.ndarray] = None


class OilStateHMM:
    def __init__(self, n_states: int = 2, max_iter: int = 100):
        self.n_states = n_states
        self.max_iter = max_iter
        self.params = HMMParameters(n_states=n_states)
        
    def _initialize_parameters(self, observations: np.ndarray):
        n = len(observations)
        
        quantiles = np.linspace(0, 1, self.n_states + 1)
        thresholds = np.quantile(observations, quantiles[1:-1])
        
        self.params.means = np.zeros(self.n_states)
        self.params.stds = np.zeros(self.n_states)
        
        for i in range(self.n_states):
            if i == 0:
                mask = observations <= thresholds[0]
            elif i == self.n_states - 1:
                mask = observations > thresholds[-1]
            else:
                mask = (observations > thresholds[i-1]) & (observations <= thresholds[i])
            
            if mask.sum() > 0:
                self.params.means[i] = observations[mask].mean()
                self.params.stds[i] = observations[mask].std()
                if self.params.stds[i] < 1e-6:
                    self.params.stds[i] = observations.std() * 0.5
        
        self.params.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        np.fill_diagonal(self.params.transition_matrix, 0.7)
        self.params.transition_matrix = self.params.transition_matrix / self.params.transition_matrix.sum(axis=1, keepdims=True)
        
        self.params.initial_probs = np.ones(self.n_states) / self.n_states
        
    def _emission_probability(self, obs: float, state: int) -> float:
        mean = self.params.means[state]
        std = self.params.stds[state]
        if std < 1e-10:
            std = 1e-10
        return stats.norm.pdf(obs, mean, std)
    
    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        T = len(observations)
        log_alpha = np.zeros((T, self.n_states))
        
        for s in range(self.n_states):
            log_alpha[0, s] = np.log(self.params.initial_probs[s] + 1e-10) + \
                             np.log(self._emission_probability(observations[0], s) + 1e-10)
        
        for t in range(1, T):
            for s in range(self.n_states):
                log_emission = np.log(self._emission_probability(observations[t], s) + 1e-10)
                log_trans = np.log(self.params.transition_matrix[:, s] + 1e-10)
                log_alpha[t, s] = log_emission + self._log_sum_exp(log_alpha[t-1] + log_trans)
        
        log_likelihood = self._log_sum_exp(log_alpha[-1])
        return np.exp(log_alpha), log_likelihood
    
    def _log_sum_exp(self, log_probs: np.ndarray) -> float:
        max_log = np.max(log_probs)
        if not np.isfinite(max_log):
            return -np.inf
        return max_log + np.log(np.sum(np.exp(log_probs - max_log)))
    
    def _backward(self, observations: np.ndarray) -> np.ndarray:
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        
        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                for next_s in range(self.n_states):
                    emission = self._emission_probability(observations[t+1], next_s)
                    beta[t, s] += (self.params.transition_matrix[s, next_s] * 
                                  emission * beta[t+1, next_s])
        
        return beta
    
    def _baum_welch_step(self, observations: np.ndarray) -> float:
        T = len(observations)
        alpha, log_likelihood = self._forward(observations)
        beta = self._backward(observations)
        
        gamma = np.zeros((T, self.n_states))
        for t in range(T):
            denominator = np.sum(alpha[t] * beta[t])
            if denominator < 1e-300:
                denominator = 1e-300
            for s in range(self.n_states):
                gamma[t, s] = (alpha[t, s] * beta[t, s]) / denominator
        
        gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            denominator = 0.0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission = self._emission_probability(observations[t+1], j)
                    xi[t, i, j] = (alpha[t, i] * self.params.transition_matrix[i, j] * 
                                  emission * beta[t+1, j])
                    denominator += xi[t, i, j]
            if denominator > 1e-300:
                xi[t] /= denominator
        
        xi = np.nan_to_num(xi, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.params.initial_probs = gamma[0]
        self.params.initial_probs = np.nan_to_num(self.params.initial_probs, nan=1.0/self.n_states)
        self.params.initial_probs /= self.params.initial_probs.sum()
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                if denominator < 1e-10:
                    self.params.transition_matrix[i, j] = 1.0 / self.n_states
                else:
                    self.params.transition_matrix[i, j] = numerator / denominator
        
        for i in range(self.n_states):
            row_sum = self.params.transition_matrix[i].sum()
            if row_sum > 0:
                self.params.transition_matrix[i] /= row_sum
            else:
                self.params.transition_matrix[i] = 1.0 / self.n_states
        
        for s in range(self.n_states):
            weights = gamma[:, s]
            weight_sum = weights.sum()
            if weight_sum < 1e-10:
                continue
            self.params.means[s] = np.sum(weights * observations) / weight_sum
            diff = observations - self.params.means[s]
            self.params.stds[s] = np.sqrt(np.sum(weights * diff**2) / weight_sum)
            if self.params.stds[s] < 1e-6:
                self.params.stds[s] = observations.std() * 0.5
        
        return log_likelihood
    
    def fit(self, observations: np.ndarray, verbose: bool = False):
        self._initialize_parameters(observations)
        
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            ll = self._baum_welch_step(observations)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Log-likelihood = {ll:.2f}")
            
            if abs(ll - prev_ll) < 1e-4:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_ll = ll
        
        if self.params.means[0] > self.params.means[1]:
            self._swap_states()
    
    def _swap_states(self):
        self.params.means = self.params.means[::-1]
        self.params.stds = self.params.stds[::-1]
        self.params.transition_matrix = self.params.transition_matrix[::-1, ::-1]
        self.params.initial_probs = self.params.initial_probs[::-1]
    
    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        for s in range(self.n_states):
            delta[0, s] = np.log(self.params.initial_probs[s] + 1e-10) + \
                         np.log(self._emission_probability(observations[0], s) + 1e-10)
        
        for t in range(1, T):
            for s in range(self.n_states):
                emission_prob = self._emission_probability(observations[t], s)
                trans_probs = delta[t-1] + np.log(self.params.transition_matrix[:, s] + 1e-10)
                psi[t, s] = np.argmax(trans_probs)
                delta[t, s] = np.max(trans_probs) + np.log(emission_prob + 1e-10)
        
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def to_dict(self) -> Dict:
        return {
            "n_states": self.n_states,
            "means": self.params.means.tolist(),
            "stds": self.params.stds.tolist(),
            "transition_matrix": self.params.transition_matrix.tolist(),
            "initial_probs": self.params.initial_probs.tolist()
        }


def prepare_oil_features(oil_df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=oil_df.index)
    
    features['oil_return'] = np.log(oil_df['near_month'] / oil_df['near_month'].shift(1))
    
    for window in [5, 10, 20]:
        features[f'return_{window}d'] = features['oil_return'].rolling(window).sum()
        features[f'vol_{window}d'] = features['oil_return'].rolling(window).std()
    
    features['slope'] = (oil_df['next_month'] - oil_df['near_month']) / oil_df['near_month']
    features['slope_ma5'] = features['slope'].rolling(5).mean()
    features['slope_ma20'] = features['slope'].rolling(20).mean()
    
    features['spread'] = oil_df['next_month'] - oil_df['near_month']
    features['spread_change'] = features['spread'].diff()
    
    return features.dropna()


def compute_gate_from_states(states: np.ndarray, hmm: OilStateHMM) -> np.ndarray:
    gate_values = np.zeros(len(states))
    
    state_means = hmm.params.means
    bear_state = np.argmin(state_means)
    bull_state = np.argmax(state_means)
    
    for i, state in enumerate(states):
        if state == bear_state:
            gate_values[i] = 0.0
        elif state == bull_state:
            gate_values[i] = 1.0
        else:
            gate_values[i] = 0.5
    
    return gate_values


def learn_oil_state_gate(oil_df: pd.DataFrame,
                         n_states: int = 2,
                         max_iter: int = 100,
                         verbose: bool = True):
    
    features = prepare_oil_features(oil_df)
    
    observations = features['oil_return'].values
    
    hmm = OilStateHMM(n_states=n_states, max_iter=max_iter)
    hmm.fit(observations, verbose=verbose)
    
    states = hmm.viterbi(observations)
    
    gate_values = compute_gate_from_states(states, hmm)
    
    results = pd.DataFrame({
        'Regime': states,
        'gate_value': gate_values,
        'oil_return': observations
    }, index=features.index)
    
    for col in features.columns:
        if col != 'oil_return':
            results[col] = features[col]
    
    return hmm, results


def plot_oil_regimes(oil_df: pd.DataFrame, results_df: pd.DataFrame, out_png: Optional[str] = None):
    df = oil_df.join(results_df, how='inner')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(df.index, df['near_month'], linewidth=1.5, label='Near Month Oil Price')
    states = df['Regime'].values
    unique_states = np.unique(states)
    
    colors = ['red', 'green', 'blue']
    for state_id in unique_states:
        mask = (states == state_id)
        if mask.sum() > 0:
            ax1.scatter(df.index[mask], df['near_month'][mask], 
                       c=colors[state_id % len(colors)], alpha=0.3, s=10,
                       label=f'State {state_id}')
    
    ax1.set_ylabel('Oil Price ($/barrel)')
    ax1.set_title('Oil Prices with HMM-Identified Regimes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df.index, df['gate_value'], linewidth=2, color='black', label='Gate Value')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(df.index, 0, df['gate_value'], alpha=0.3)
    ax2.set_ylabel('Gate Value (0-1)')
    ax2.set_xlabel('Date')
    ax2.set_title('Oil State Gate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    return fig


def run_oil_state_gate(oil_csv: str,
                       out_gates_csv: str = "oil_gates_hmm.csv",
                       out_hmm_json: str = "oil_hmm_params.json",
                       out_plot_png: Optional[str] = "oil_state_plot.png",
                       n_states: int = 2,
                       max_iter: int = 100):
    
    oil_df = pd.read_csv(oil_csv)
    date_col = "time" if "time" in oil_df.columns else "date"
    oil_df[date_col] = pd.to_datetime(oil_df[date_col])
    oil_df = oil_df.set_index(date_col).sort_index()
    
    print(f"Loaded oil data: {len(oil_df)} days")
    print(f"Date range: {oil_df.index.min().date()} to {oil_df.index.max().date()}")
    
    hmm, results = learn_oil_state_gate(oil_df, n_states=n_states, 
                                        max_iter=max_iter, verbose=True)
    
    print("\n" + "="*60)
    print("HMM PARAMETERS")
    print("="*60)
    print(f"State means (returns): {hmm.params.means}")
    print(f"State std devs: {hmm.params.stds}")
    print(f"Transition matrix:\n{hmm.params.transition_matrix}")
    
    state_labels = ['Bear (Low Return)', 'Bull (High Return)']
    if n_states > 2:
        state_labels = [f'State {i}' for i in range(n_states)]
    
    print("\n" + "="*60)
    print("REGIME STATISTICS")
    print("="*60)
    for state_id in range(n_states):
        mask = results['Regime'] == state_id
        count = mask.sum()
        pct = 100 * count / len(results)
        mean_ret = results.loc[mask, 'oil_return'].mean()
        print(f"{state_labels[state_id]:20s}: {count:5d} days ({pct:5.1f}%), Mean return: {mean_ret:+.4f}")
    
    results_out = results[['Regime', 'gate_value']].copy()
    results_out.index.name = 'Date'
    results_out.to_csv(out_gates_csv)
    print(f"\nSaved gates: {out_gates_csv}")
    
    with open(out_hmm_json, "w") as f:
        json.dump(hmm.to_dict(), f, indent=2)
    print(f"Saved HMM parameters: {out_hmm_json}")
    
    try:
        plot_oil_regimes(oil_df, results, out_plot_png)
        print(f"Saved plot: {out_plot_png}")
    except Exception as e:
        print(f"Plotting skipped: {e}")
    
    return hmm, results


def cli():
    parser = argparse.ArgumentParser(description="Oil state gate using HMM")
    parser.add_argument("--oil_csv", required=True, help="CSV with oil prices (near_month, next_month)")
    parser.add_argument("--out_gates_csv", default="oil_gates_hmm.csv", help="Output gates CSV")
    parser.add_argument("--out_hmm_json", default="oil_hmm_params.json", help="Output HMM parameters")
    parser.add_argument("--plot_png", default="oil_state_plot.png", help="Output plot")
    parser.add_argument("--n_states", type=int, default=2, help="Number of HMM states")
    parser.add_argument("--max_iter", type=int, default=100, help="Max HMM iterations")
    args = parser.parse_args()
    
    run_oil_state_gate(
        oil_csv=args.oil_csv,
        out_gates_csv=args.out_gates_csv,
        out_hmm_json=args.out_hmm_json,
        out_plot_png=args.plot_png,
        n_states=args.n_states,
        max_iter=args.max_iter
    )


if __name__ == "__main__":
    cli()