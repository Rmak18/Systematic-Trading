import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# Add gates directory to path (go up from scripts/test/ to root, then into gates/)
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
gates_path = os.path.join(repo_root, 'gates')
sys.path.insert(0, gates_path)

from oil_state_gate_hmm import OilStateGateHMM, prepare_oil_data


class HMMValidator:
    """Comprehensive validation suite for HMM stability and robustness."""
    
    def __init__(self, oil_csv: str, train_end: str = "2018-12-31"):
        self.oil_csv = oil_csv
        self.train_end = train_end
        self.results = {}
        
    def load_and_prepare_data(self, use_features: List[str] = None):
        """Load and prepare oil data."""
        if use_features is None:
            use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
        
        oil_df = pd.read_csv(self.oil_csv)
        date_col = "time" if "time" in oil_df.columns else "date"
        oil_df[date_col] = pd.to_datetime(oil_df[date_col])
        oil_df = oil_df.set_index(date_col).sort_index()
        
        features_df, features_array, feature_names = prepare_oil_data(oil_df, use_features)
        
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        test_features = features_array[~train_mask]
        
        return features_df, train_features, test_features, feature_names
    
    def test_seed_stability(self, n_seeds: int = 20, n_states: int = 2):
        """Test 1: How sensitive is the model to random initialization?"""
        print("\n" + "="*70)
        print("TEST 1: SEED STABILITY")
        print("="*70)
        print(f"Training {n_seeds} models with different random seeds...")
        
        features_df, train_features, test_features, feature_names = self.load_and_prepare_data()
        
        seed_results = []
        
        for seed in range(n_seeds):
            hmm = OilStateGateHMM(n_states=n_states, random_state=seed, n_iter=100)
            hmm.fit(train_features, feature_names=feature_names, verbose=False)
            
            # Get test period predictions
            test_states = hmm.predict_states(test_features)
            test_probs = hmm.predict_probabilities(test_features)
            
            # Calculate metrics
            bear_days = (test_states == hmm.bear_state).sum()
            bull_days = (test_states == hmm.bull_state).sum()
            
            bear_mean = hmm.model.means_[hmm.bear_state, 0]
            bull_mean = hmm.model.means_[hmm.bull_state, 0]
            
            bear_persist = hmm.model.transmat_[hmm.bear_state, hmm.bear_state]
            bull_persist = hmm.model.transmat_[hmm.bull_state, hmm.bull_state]
            
            seed_results.append({
                'seed': seed,
                'bear_mean': bear_mean,
                'bull_mean': bull_mean,
                'bear_persist': bear_persist,
                'bull_persist': bull_persist,
                'bear_days_pct': 100 * bear_days / len(test_states),
                'bull_days_pct': 100 * bull_days / len(test_states),
                'log_likelihood': hmm.model.score(train_features)
            })
        
        seed_df = pd.DataFrame(seed_results)
        
        print("\nSummary Statistics Across Seeds:")
        print(seed_df.describe().round(4))
        
        print("\nStability Metrics:")
        print(f"  Bear mean range: [{seed_df['bear_mean'].min():.6f}, {seed_df['bear_mean'].max():.6f}]")
        print(f"  Bull mean range: [{seed_df['bull_mean'].min():.6f}, {seed_df['bull_mean'].max():.6f}]")
        print(f"  Bear persist std: {seed_df['bear_persist'].std():.4f}")
        print(f"  Bull persist std: {seed_df['bull_persist'].std():.4f}")
        
        # Verdict
        bear_mean_std = seed_df['bear_mean'].std()
        bull_mean_std = seed_df['bull_mean'].std()
        
        if bear_mean_std < 0.0005 and bull_mean_std < 0.0005:
            print("\n✅ VERDICT: Model is VERY STABLE across seeds")
            print("   → PSO for seed optimization is NOT needed")
        elif bear_mean_std < 0.001 and bull_mean_std < 0.001:
            print("\n✓ VERDICT: Model is reasonably stable")
            print("   → Minor variation exists but likely not problematic")
        else:
            print("\n⚠️  VERDICT: Model shows significant seed sensitivity")
            print("   → Consider ensemble approach or seed optimization")
        
        self.results['seed_stability'] = seed_df
        return seed_df
    
    def test_feature_importance(self, n_states: int = 2, seed: int = 42):
        """Test 2: Which features actually matter?"""
        print("\n" + "="*70)
        print("TEST 2: FEATURE IMPORTANCE (Leave-One-Out)")
        print("="*70)
        
        all_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
        
        # Baseline: All features
        print("\nBaseline (all features):")
        features_df, train_features, test_features, feature_names = self.load_and_prepare_data(all_features)
        
        baseline_hmm = OilStateGateHMM(n_states=n_states, random_state=seed, n_iter=100)
        baseline_hmm.fit(train_features, feature_names=feature_names, verbose=False)
        baseline_ll = baseline_hmm.model.score(train_features)
        
        baseline_test_states = baseline_hmm.predict_states(test_features)
        baseline_bear_pct = 100 * (baseline_test_states == baseline_hmm.bear_state).sum() / len(test_features)
        
        print(f"  Log-likelihood: {baseline_ll:.2f}")
        print(f"  Bear days (test): {baseline_bear_pct:.1f}%")
        
        # Leave-one-out
        feature_importance = []
        
        for exclude_feature in all_features:
            remaining_features = [f for f in all_features if f != exclude_feature]
            print(f"\nWithout {exclude_feature}: {remaining_features}")
            
            features_df, train_features, test_features, feature_names = self.load_and_prepare_data(remaining_features)
            
            hmm = OilStateGateHMM(n_states=n_states, random_state=seed, n_iter=100)
            hmm.fit(train_features, feature_names=feature_names, verbose=False)
            ll = hmm.model.score(train_features)
            
            test_states = hmm.predict_states(test_features)
            bear_pct = 100 * (test_states == hmm.bear_state).sum() / len(test_features)
            
            ll_diff = baseline_ll - ll
            bear_diff = baseline_bear_pct - bear_pct
            
            print(f"  Log-likelihood: {ll:.2f} (Δ = {ll_diff:+.2f})")
            print(f"  Bear days (test): {bear_pct:.1f}% (Δ = {bear_diff:+.1f}%)")
            
            feature_importance.append({
                'excluded_feature': exclude_feature,
                'log_likelihood': ll,
                'll_drop': ll_diff,
                'bear_pct': bear_pct,
                'bear_pct_change': bear_diff
            })
        
        importance_df = pd.DataFrame(feature_importance)
        importance_df = importance_df.sort_values('ll_drop', ascending=False)
        
        print("\n" + "="*70)
        print("Feature Importance Ranking (by log-likelihood drop):")
        print("="*70)
        for _, row in importance_df.iterrows():
            print(f"  {row['excluded_feature']:15s}: LL drop = {row['ll_drop']:+7.2f} "
                  f"(Bear% Δ = {row['bear_pct_change']:+5.1f}%)")
        
        # Verdict
        max_drop = importance_df['ll_drop'].max()
        if max_drop < 50:
            print("\n✅ VERDICT: All features contribute similarly")
            print("   → No clear feature to drop, feature weighting unlikely to help much")
        elif max_drop < 200:
            print("\n✓ VERDICT: Some features more important than others")
            print("   → Could benefit from feature selection or weighting")
        else:
            print("\n⚠️  VERDICT: Some features are MUCH more important")
            print("   → Consider dropping weak features or using weighted approach")
        
        self.results['feature_importance'] = importance_df
        return importance_df
    
    def test_walk_forward_validation(self, n_splits: int = 5, n_states: int = 2, seed: int = 42):
        """Test 3: Walk-forward validation (does it generalize?)"""
        print("\n" + "="*70)
        print("TEST 3: WALK-FORWARD VALIDATION")
        print("="*70)
        print(f"Testing {n_splits} train/test splits...\n")
        
        features_df, _, _, feature_names = self.load_and_prepare_data()
        full_features = features_df[['oil_return', 'slope', 'vol_20d', 'return_20d']].values
        
        # Create expanding window splits
        total_len = len(features_df)
        min_train = int(0.5 * total_len)  # Start with 50% for training
        split_points = np.linspace(min_train, int(0.9 * total_len), n_splits, dtype=int)
        
        wf_results = []
        
        for i, split_point in enumerate(split_points):
            train_data = full_features[:split_point]
            test_data = full_features[split_point:split_point + 200]  # 200 days test
            
            if len(test_data) < 50:
                continue
            
            train_dates = features_df.index[:split_point]
            test_dates = features_df.index[split_point:split_point + 200]
            
            print(f"Split {i+1}: Train={train_dates[0].date()} to {train_dates[-1].date()} "
                  f"({len(train_data)} days), Test={test_dates[0].date()} to {test_dates[-1].date()} "
                  f"({len(test_data)} days)")
            
            hmm = OilStateGateHMM(n_states=n_states, random_state=seed, n_iter=100)
            hmm.fit(train_data, feature_names=feature_names, verbose=False)
            
            # Test period metrics
            test_states = hmm.predict_states(test_data)
            bear_pct = 100 * (test_states == hmm.bear_state).sum() / len(test_states)
            
            bear_mean = hmm.model.means_[hmm.bear_state, 0]
            bull_mean = hmm.model.means_[hmm.bull_state, 0]
            
            print(f"  Bear mean: {bear_mean:+.6f}, Bull mean: {bull_mean:+.6f}")
            print(f"  Test bear days: {bear_pct:.1f}%\n")
            
            wf_results.append({
                'split': i+1,
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'bear_mean': bear_mean,
                'bull_mean': bull_mean,
                'bear_pct': bear_pct
            })
        
        wf_df = pd.DataFrame(wf_results)
        
        print("="*70)
        print("Walk-Forward Summary:")
        print("="*70)
        print(f"  Bear mean: {wf_df['bear_mean'].mean():.6f} ± {wf_df['bear_mean'].std():.6f}")
        print(f"  Bull mean: {wf_df['bull_mean'].mean():.6f} ± {wf_df['bull_mean'].std():.6f}")
        print(f"  Bear days %: {wf_df['bear_pct'].mean():.1f}% ± {wf_df['bear_pct'].std():.1f}%")
        
        # Verdict
        bear_mean_std = wf_df['bear_mean'].std()
        bull_mean_std = wf_df['bull_mean'].std()
        
        if bear_mean_std < 0.001 and bull_mean_std < 0.001:
            print("\n✅ VERDICT: Model generalizes WELL across time periods")
            print("   → Stable and consistent, optimization unlikely to help much")
        elif bear_mean_std < 0.002 and bull_mean_std < 0.002:
            print("\n✓ VERDICT: Reasonable consistency across time")
            print("   → Some variation but acceptable")
        else:
            print("\n⚠️  VERDICT: Model shows time-period sensitivity")
            print("   → Results vary significantly across time, consider regime-adaptive approach")
        
        self.results['walk_forward'] = wf_df
        return wf_df
    
    def generate_report(self, save_path: str = "hmm_validation_report.txt"):
        """Generate comprehensive validation report."""
        print("\n" + "="*70)
        print("VALIDATION REPORT SUMMARY")
        print("="*70)
        
        report = []
        report.append("HMM VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall recommendation
        report.append("OVERALL RECOMMENDATION:")
        report.append("-" * 70)
        
        needs_optimization = False
        reasons = []
        
        if 'seed_stability' in self.results:
            seed_df = self.results['seed_stability']
            if seed_df['bear_mean'].std() > 0.001:
                needs_optimization = True
                reasons.append("- High seed sensitivity detected")
        
        if 'feature_importance' in self.results:
            imp_df = self.results['feature_importance']
            if imp_df['ll_drop'].max() > 200:
                reasons.append("- Some features much more important than others")
        
        if 'walk_forward' in self.results:
            wf_df = self.results['walk_forward']
            if wf_df['bear_mean'].std() > 0.002:
                needs_optimization = True
                reasons.append("- Time-period sensitivity detected")
        
        if needs_optimization:
            report.append("⚠️  OPTIMIZATION MAY BE BENEFICIAL")
            report.append("")
            report.append("Issues found:")
            for reason in reasons:
                report.append(f"  {reason}")
            report.append("")
            report.append("Recommended approaches:")
            report.append("  1. Ensemble multiple random seeds (simple, fast)")
            report.append("  2. Feature selection (remove weak features)")
            report.append("  3. If still needed: PSO for feature weighting")
        else:
            report.append("✅ MODEL IS STABLE - OPTIMIZATION NOT NEEDED")
            report.append("")
            report.append("Your current HMM is:")
            report.append("  - Stable across random seeds")
            report.append("  - Features contribute meaningfully")
            report.append("  - Generalizes across time periods")
            report.append("")
            report.append("RECOMMENDATION: Focus on trading strategy, not model optimization")
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {save_path}")


def run_full_validation(oil_csv: str, train_end: str = "2018-12-31"):
    """Run all validation tests."""
    validator = HMMValidator(oil_csv, train_end)
    
    # Test 1: Seed stability
    validator.test_seed_stability(n_seeds=20)
    
    # Test 2: Feature importance
    validator.test_feature_importance()
    
    # Test 3: Walk-forward validation
    validator.test_walk_forward_validation(n_splits=5)
    
    # Generate report
    validator.generate_report()
    
    return validator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate HMM model stability and robustness")
    parser.add_argument("--oil_csv", required=True, help="Path to oil prices CSV")
    parser.add_argument("--train_end", default="2018-12-31", help="Training period end date")
    args = parser.parse_args()
    
    run_full_validation(args.oil_csv, args.train_end)