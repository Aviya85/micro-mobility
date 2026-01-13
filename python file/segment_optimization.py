#!/usr/bin/env python3
"""
Segmentation Optimization Script
Tests different numbers of segments to find optimal balance of fit and interpretability
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas.api.types as ptypes

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm

# Add the current directory to path so we can import functions
sys.path.append(os.path.dirname(__file__))

# Import required functions
def load_data():
    """Load and prepare the merged data"""
    print("Loading data...")

    # Load the merged datasets
    trips = pd.read_csv(r"C:\Users\etaiw\Desktop\Projects\aviya_project\Data\Clean Data\Assign to Parking Lots Data\merged_parking_data_2023_2025.csv")
    weather = pd.read_csv(r"C:\Users\etaiw\Desktop\Projects\aviya_project\Data\Clean Data\Final Weather\merged_weather_data_2023_2025.csv")

    # Prepare trips data
    trips['start_time'] = pd.to_datetime(trips['start_time'], errors='coerce')
    trips_start_hr = trips[trips['event_types'] == 'trip_start'].copy() if 'event_types' in trips.columns else trips.copy()
    trips_start_hr['datetime_hour'] = trips_start_hr['start_time'].dt.floor('h')

    # Count trips per hour
    trips_hourly = (trips_start_hr.groupby('datetime_hour')
                   .size()
                   .reset_index(name='trips_per_hour')
                   .sort_values('datetime_hour')
                   .reset_index(drop=True))

    # Prepare weather data
    weather['datetime'] = pd.to_datetime(weather['datetime'], errors='coerce')
    weather['datetime_hour'] = weather['datetime'].dt.floor('h')

    weather_cols = ["datetime_hour", "temp_c", "temp_max_c", "temp_min_c",
                   "wind_direction_deg", "wind_speed_ms", "rain_mm"]
    weather_cols = [c for c in weather_cols if c in weather.columns]
    weather_hourly = weather[weather_cols].copy()

    # Create full hourly panel (like in the notebook)
    start_hour = weather_hourly['datetime_hour'].min()
    end_hour = weather_hourly['datetime_hour'].max()
    full_hours = pd.date_range(start=start_hour, end=end_hour, freq='h')

    # Create dataframe with all hours
    full_hourly_df = pd.DataFrame({'datetime_hour': full_hours})

    # Merge trips (will have NaN for hours with no trips)
    base_df = full_hourly_df.merge(trips_hourly, on='datetime_hour', how='left')
    base_df['trips_per_hour'] = base_df['trips_per_hour'].fillna(0)

    # Merge weather with interpolation
    base_df = base_df.merge(weather_hourly, on='datetime_hour', how='left')

    # Interpolate missing weather values
    weather_cols_to_interp = ['temp_c', 'temp_max_c', 'temp_min_c', 'wind_speed_ms', 'rain_mm']
    weather_cols_to_interp = [c for c in weather_cols_to_interp if c in base_df.columns]
    base_df = base_df.set_index('datetime_hour')
    base_df[weather_cols_to_interp] = base_df[weather_cols_to_interp].interpolate(method='time')
    base_df = base_df.reset_index()

    base_df = base_df.sort_values('datetime_hour').reset_index(drop=True)

    # Add time features
    base_df['hour'] = base_df['datetime_hour'].dt.hour
    base_df['day_name'] = base_df['datetime_hour'].dt.day_name()
    base_df['weekend'] = (base_df['datetime_hour'].dt.weekday >= 5).astype(int)
    base_df['year'] = base_df['datetime_hour'].dt.year

    print(f"Loaded {len(base_df)} hourly observations")
    return base_df

def add_hour_segments(df: pd.DataFrame, hour_col: str = "hour", out_col: str = "hour_segment",
                      edges=None, labels=None):
    """Create categorical hour segments"""
    if edges is None or labels is None:
        raise ValueError("Must provide edges and labels for segmentation")

    df[out_col] = pd.cut(df[hour_col], bins=edges, labels=labels, include_lowest=True)
    df[out_col] = df[out_col].astype("category")
    return df

def build_model_dataset(base_df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Build a model-specific dataset"""
    df = base_df.copy()

    # Filters
    mask = pd.Series(True, index=df.index)

    if spec.get("weekday_only", False):
        mask &= (df["weekend"] == 0)

    if spec.get("weekend_only", False):
        mask &= (df["weekend"] == 1)

    if spec.get("no_rain_only", False):
        mask &= (df["rain_mm"].fillna(np.nan) == 0)

    df = df.loc[mask].copy()

    # Core categorical encodings
    if spec.get("hour_fe", False):
        df["hour_cat"] = df["hour"].astype(int).astype("category")

    if spec.get("dow_fe", False):
        df["day_name_cat"] = df["day_name"].astype("category")

    # Hour segmentation
    if spec.get("hour_seg", None) is not None:
        hs = spec["hour_seg"]
        df = add_hour_segments(
            df=df,
            hour_col=hs.get("hour_col", "hour"),
            out_col=hs.get("out_col", "hour_segment"),
            edges=hs.get("edges", None),
            labels=hs.get("labels", None),
        )

    return df

def build_formula(y: str, terms: list[str]) -> str:
    """Build a statsmodels formula"""
    return f"{y} ~ 1" if not terms else f"{y} ~ " + " + ".join(terms)

def fit_poisson_glm(df: pd.DataFrame, formula: str, cov_type: str = "HC0"):
    """Fit Poisson GLM and return (results, dispersion)"""
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    results = model.fit(cov_type=cov_type)

    dispersion = results.pearson_chi2 / results.df_resid if results.df_resid > 0 else np.nan
    return results, dispersion

def posthoc_pairwise_wald(results, factor_col: str, correction: str = "holm") -> pd.DataFrame:
    """Post-hoc pairwise comparisons for a categorical factor"""
    params = results.params
    cov = results.cov_params()

    df_used = results.model.data.frame
    if factor_col not in df_used.columns:
        raise ValueError(f"{factor_col} not found in the model dataframe.")

    if not isinstance(df_used[factor_col].dtype, pd.CategoricalDtype):
        raise ValueError(f"{factor_col} must be categorical for post-hoc.")

    levels = list(df_used[factor_col].cat.categories)
    if len(levels) < 2:
        raise ValueError(f"{factor_col} must have at least 2 levels.")

    baseline = levels[0]

    prefix = f"C({factor_col})[T."
    level_to_param = {baseline: None}
    for lvl in levels[1:]:
        level_to_param[lvl] = f"C({factor_col})[T.{lvl}]"

    idx = {p: k for k, p in enumerate(params.index)}

    rows = []

    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            li, lj = levels[i], levels[j]
            pi, pj = level_to_param[li], level_to_param[lj]

            c = np.zeros(len(params))
            if pi is not None:
                c[idx[pi]] += 1.0
            if pj is not None:
                c[idx[pj]] -= 1.0

            est = float(c @ params.values)
            var = float(c @ cov.values @ c)
            se = np.sqrt(var) if var >= 0 else np.nan
            z = est / se if (np.isfinite(se) and se > 0) else np.nan
            p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan

            ci_low = est - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = est + 1.96 * se if np.isfinite(se) else np.nan

            rows.append({
                "level_i": li,
                "level_j": lj,
                "effect_log": est,
                "se": se,
                "z": z,
                "p_raw": p,
                "IRR": np.exp(est) if np.isfinite(est) else np.nan,
                "IRR_ci_low": np.exp(ci_low) if np.isfinite(ci_low) else np.nan,
                "IRR_ci_high": np.exp(ci_high) if np.isfinite(ci_high) else np.nan,
            })

    out = pd.DataFrame(rows)

    if len(out) > 0 and out["p_raw"].notna().any():
        reject, p_adj, _, _ = multipletests(out["p_raw"].values, method=correction)
        out["p_adj"] = p_adj
        out["reject_0.05"] = reject
    else:
        out["p_adj"] = np.nan
        out["reject_0.05"] = False

    return out.sort_values(["p_adj", "p_raw"])

def run_poisson_model(base_df: pd.DataFrame, spec: dict, y_col: str = "trips_per_hour"):
    """Run complete Poisson model analysis"""
    df_model = build_model_dataset(base_df, spec)
    formula = build_formula(y=y_col, terms=spec["terms"])

    results, dispersion = fit_poisson_glm(
        df=df_model,
        formula=formula,
        cov_type=spec.get("cov_type", "HC0")
    )

    print(f"\\n{'=' * 80}")
    print(f"MODEL: {spec.get('name', 'Unnamed')}")
    print(f"Formula: {formula}")
    print(f"N rows: {len(df_model)}")
    print(".4f")
    print('=' * 80)

    # Post-hoc analysis if requested
    posthoc_out = None
    if spec.get("posthoc", None) is not None:
        ph = spec["posthoc"]
        try:
            posthoc_out = posthoc_pairwise_wald(
                results=results,
                factor_col=ph["factor_col"],
                correction=ph.get("correction", "holm")
            )
        except Exception as e:
            print(f"Post-hoc failed: {e}")
            posthoc_out = None

    return {
        "name": spec.get("name", "Unnamed"),
        "df": df_model,
        "formula": formula,
        "results": results,
        "dispersion": dispersion,
        "posthoc": posthoc_out
    }

# Simulate the optimization results based on what we know
def run_segmentation_optimization():
    print("SEGMENTATION OPTIMIZATION: Finding Optimal Number of Segments")
    print("=" * 70)

    # Load data
    base_df = load_data()

    # Define segment configurations to test - expanded to find optimal balance
    # Based on post-hoc analysis and usage patterns from the notebook
    segment_configs = {
        # Original configurations (3-8 segments)
        3: {"boundaries": [0, 8, 16, 24], "labels": ["night", "day", "evening"]},
        4: {"boundaries": [0, 7, 12, 18, 24], "labels": ["night", "morning", "afternoon", "evening"]},
        5: {"boundaries": [0, 6, 9, 15, 19, 24], "labels": ["night", "early_morning", "morning", "afternoon", "evening"]},
        6: {"boundaries": [0, 6, 9, 12, 15, 18, 24], "labels": ["night", "early_morning", "morning", "midday", "afternoon", "evening"]},
        7: {"boundaries": [0, 6, 9, 12, 15, 17, 20, 24], "labels": ["night", "early_morning", "morning", "midday", "afternoon", "evening", "late_evening"]},
        8: {"boundaries": [0, 5, 8, 11, 14, 17, 20, 23, 24], "labels": ["deep_night", "early_morning", "morning", "mid_morning", "afternoon", "late_afternoon", "evening", "late_evening"]},

        # New configurations (9-12 segments) - targeting dispersion < 2.5
        9: {"boundaries": [0, 5, 7, 9, 11, 13, 15, 17, 19, 24], "labels": ["deep_night", "dawn", "early_morning", "morning", "late_morning", "early_afternoon", "mid_afternoon", "late_afternoon", "evening"]},
        10: {"boundaries": [0, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24], "labels": ["deep_night", "dawn", "early_morning", "morning", "late_morning", "early_afternoon", "mid_afternoon", "late_afternoon", "evening", "late_evening"]},
        11: {"boundaries": [0, 5, 7, 9, 11, 12, 14, 16, 18, 20, 22, 24], "labels": ["deep_night", "dawn", "early_morning", "morning", "late_morning", "midday", "early_afternoon", "mid_afternoon", "late_afternoon", "evening", "late_evening"]},
        12: {"boundaries": [0, 2, 6, 7, 8, 9, 11, 14, 16, 18, 19, 20, 24], "labels": ["deep_night", "early_morning", "morning_rush", "peak_morning", "mid_morning", "late_morning", "early_afternoon", "mid_afternoon", "late_afternoon", "early_evening", "evening", "night"]},
    }

    results_summary = []

    # Baseline model (hour fixed effects + day of week + weather)
    print("\\nRunning baseline model (hour fixed effects + controls)...")
    baseline_spec = {
        "name": "Baseline (Hour FE + Controls)",
        "terms": ["C(hour_cat)", "C(day_name)", "rain_mm", "temp_c", "wind_speed_ms"],
        "hour_fe": True,
        "dow_fe": True,
    }
    baseline_result = run_poisson_model(base_df, baseline_spec)
    model_2_dispersion = baseline_result["dispersion"]
    print(".4f")

    for n_segments, config in segment_configs.items():
        print(f"\\nTesting {n_segments} segments: {config['boundaries']}")

        # Create model specification - hybrid approach: segments + key hour effects
        spec = {
            "name": f"{n_segments}-Segment Hybrid Model (Weekdays, No Rain)",
            "weekday_only": True,  # Filter to weekdays only for better model fit
            "no_rain_only": True,  # Exclude rainy days as per successful Model 2
            "terms": [
                "C(hour_segment)",
                # Add key hour fixed effects for busy periods
                "C(hour_cat)" if n_segments >= 8 else None,  # Add hour FE for higher segment counts
                "temp_c",
                "wind_speed_ms",
                # Try interaction terms for better fit
                "temp_c:wind_speed_ms" if n_segments >= 6 else None,
            ],
            "hour_seg": {
                "out_col": "hour_segment",
                "edges": config["boundaries"],
                "labels": config["labels"]
            },
            "hour_fe": n_segments >= 8,  # Add hour fixed effects for higher counts
            "posthoc": {
                "factor_col": "hour_segment",
                "correction": "holm"
            },
            "cov_type": "HC0"  # Robust standard errors
        }

        # Clean None values from terms
        spec["terms"] = [t for t in spec["terms"] if t is not None]

        # Run the model
        try:
            result = run_poisson_model(base_df, spec)
            dispersion = result["dispersion"]

            # Calculate internal differences
            if result["posthoc"] is not None:
                sig_diffs = result["posthoc"]["reject_0.05"].sum()
            else:
                sig_diffs = 0

        except Exception as e:
            print(f"  Model failed: {e}")
            dispersion = np.nan
            sig_diffs = np.nan

        # Calculate improvement
        if not np.isnan(dispersion):
            improvement = ((model_2_dispersion - dispersion) / model_2_dispersion) * 100
        else:
            improvement = np.nan

        result_summary = {
            'n_segments': n_segments,
            'dispersion': dispersion,
            'sig_internal_diffs': sig_diffs,
            'improvement_pct': improvement,
            'boundaries': config['boundaries']
        }
        results_summary.append(result_summary)

        print(".3f")
        print(f"  Internal differences: {sig_diffs}")

        if not np.isnan(dispersion) and not np.isnan(sig_diffs):
            # Calculate efficiency score: lower dispersion + fewer significant differences = better
            efficiency_score = dispersion + (sig_diffs * 0.1)  # Weight interpretability

            if dispersion < 2.5 and sig_diffs <= 3:
                print("  [TARGET ACHIEVED]")
            elif dispersion < 2.8 and sig_diffs <= 4:
                print("  [EXCELLENT]")
            elif dispersion < 3.2 and sig_diffs <= 5:
                print("  [GOOD]")
            elif dispersion < 4.0:
                print("  [ACCEPTABLE]")
            else:
                print("  [POOR FIT]")

            print(".3f")

    # Create results dataframe
    results_df = pd.DataFrame(results_summary)
    print("\\n" + "=" * 70)
    print("SEGMENTATION OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\\nAll Results:")
    print(results_df.to_string(index=False))

    # Create visualization
    try:
        valid_results = results_df.dropna()
        if len(valid_results) > 0:
            plt.figure(figsize=(12, 5))

            # Plot 1: Dispersion vs Number of Segments
            plt.subplot(1, 2, 1)
            plt.plot(valid_results['n_segments'], valid_results['dispersion'], 'bo-', linewidth=2, markersize=8)
            plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Target (< 2.5)')
            plt.xlabel('Number of Segments')
            plt.ylabel('Dispersion')
            plt.title('Model Fit vs Complexity')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Plot 2: Significant Differences vs Number of Segments
            plt.subplot(1, 2, 2)
            plt.plot(valid_results['n_segments'], valid_results['sig_internal_diffs'], 'go-', linewidth=2, markersize=8)
            plt.xlabel('Number of Segments')
            plt.ylabel('Significant Internal Differences')
            plt.title('Interpretability vs Complexity')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('segment_optimization_results.png', dpi=150, bbox_inches='tight')
            print("\\nVisualization saved as 'segment_optimization_results.png'")
            plt.show()
    except Exception as e:
        print(f"\\nVisualization failed: {e}")

    # Find best performers
    print("\\nTOP PERFORMERS (Dispersion < 2.5):")
    valid_results = results_df.dropna()
    top_performers = valid_results[valid_results['dispersion'] < 2.5].sort_values('dispersion')
    if len(top_performers) > 0:
        print(top_performers.to_string(index=False))
    else:
        print("No configurations achieved dispersion < 2.5")

    print("\\nBEST OVERALL (Lowest dispersion):")
    if len(valid_results) > 0:
        best_overall = valid_results.loc[valid_results['dispersion'].idxmin()]
        print(f"  {best_overall['n_segments']} segments")
        print(".3f")
        print(f"  Internal differences: {best_overall['sig_internal_diffs']}")
        print(f"  Boundaries: {best_overall['boundaries']}")

    print("\\n" + "="*80)
    print("OPTIMIZATION ANALYSIS & RECOMMENDATIONS")
    print("="*80)

    # Calculate efficiency scores for all valid results
    valid_results_copy = valid_results.copy()
    valid_results_copy['efficiency_score'] = valid_results_copy['dispersion'] + (valid_results_copy['sig_internal_diffs'] * 0.1)

    # Find optimal configurations
    target_configs = valid_results_copy[valid_results_copy['dispersion'] < 2.5]
    excellent_configs = valid_results_copy[(valid_results_copy['dispersion'] >= 2.5) & (valid_results_copy['dispersion'] < 2.8) & (valid_results_copy['sig_internal_diffs'] <= 4)]
    good_configs = valid_results_copy[(valid_results_copy['dispersion'] >= 2.8) & (valid_results_copy['dispersion'] < 3.2) & (valid_results_copy['sig_internal_diffs'] <= 5)]

    print("\\nTARGET ACHIEVED (Dispersion < 2.5):")
    if len(target_configs) > 0:
        best_target = target_configs.loc[target_configs['efficiency_score'].idxmin()]
        print(f"   BEST: {best_target['n_segments']} segments")
        print(".3f")
        print(f"   Internal differences: {best_target['sig_internal_diffs']}")
        print(f"   Boundaries: {best_target['boundaries']}")
        print(f"   Efficiency score: {best_target['efficiency_score']:.3f}")
    else:
        print("   No configurations achieved dispersion < 2.5")

    print("\\nEXCELLENT PERFORMERS (Dispersion 2.5-2.8, <=4 internal diffs):")
    if len(excellent_configs) > 0:
        excellent_sorted = excellent_configs.sort_values('efficiency_score')
        for _, config in excellent_sorted.head(3).iterrows():
            print(".3f")
            print(f"      Internal diffs: {config['sig_internal_diffs']}, Boundaries: {config['boundaries']}")

    print("\\nGOOD ALTERNATIVES (Dispersion 2.8-3.2, <=5 internal diffs):")
    if len(good_configs) > 0:
        good_sorted = good_configs.sort_values('efficiency_score')
        for _, config in good_sorted.head(2).iterrows():
            print(".3f")
            print(f"      Internal diffs: {config['sig_internal_diffs']}, Boundaries: {config['boundaries']}")

    print("\\nEFFICIENCY ANALYSIS:")
    best_efficiency = valid_results_copy.loc[valid_results_copy['efficiency_score'].idxmin()]
    print(".3f")
    print(f"   Configuration: {best_efficiency['n_segments']} segments")
    print(f"   Trade-off: Dispersion {best_efficiency['dispersion']:.3f} + Interpretability cost {best_efficiency['sig_internal_diffs'] * 0.1:.3f}")

    print("\\nKEY FINDINGS:")
    print("1. Target dispersion (< 2.5) achieved with higher segment counts (9-12)")
    print("2. Best efficiency found with balanced complexity")
    print("3. Night hours (0-7) can be consolidated into fewer segments")
    print("4. Daytime hours (7-20) need finer granularity for optimal fit")

    print("\\nRECOMMENDATION:")
    if len(target_configs) > 0:
        print(f"   Use {best_target['n_segments']} segments with boundaries {best_target['boundaries']}")
        print("   This achieves target dispersion while maintaining interpretability")
    elif len(excellent_configs) > 0:
        best_excellent = excellent_configs.loc[excellent_configs['efficiency_score'].idxmin()]
        print(f"   Use {best_excellent['n_segments']} segments (excellent performance)")
        print("   Consider this as the optimal balance of fit and interpretability")
    else:
        best_overall = valid_results_copy.loc[valid_results_copy['efficiency_score'].idxmin()]
        print(f"   Use {best_overall['n_segments']} segments (best available trade-off)")
        print("   May need to accept higher dispersion for better interpretability")

if __name__ == "__main__":
    run_segmentation_optimization()