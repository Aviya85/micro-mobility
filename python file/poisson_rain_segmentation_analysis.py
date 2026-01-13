# Poisson Regression Rain Segmentation Analysis
# Complete analysis based on optimal 12-segment model from 5.1

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas.api.types as ptypes
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("Starting Poisson Regression Rain Segmentation Analysis...")
print("=" * 70)

# =================================================
# DATA LOADING AND PREPARATION
# =================================================

print("Step 1: Loading data...")
status_final = pd.read_csv(r"C:\Users\etaiw\Desktop\Projects\aviya_project\Data\Clean Data\Assign to Parking Lots Data\merged_parking_data_2023_2025.csv")
weather_original = pd.read_csv(r"C:\Users\etaiw\Desktop\Projects\aviya_project\Data\Clean Data\Final Weather\merged_weather_data_2023_2025.csv")

trips = status_final
weather = weather_original

print(f"Data loaded - trips: {trips.shape}, weather: {weather.shape}")

# Prepare trips data
print("Step 2: Preparing trips data...")
trips['start_time'] = pd.to_datetime(trips['start_time'], errors='coerce')

if 'event_types' in trips.columns:
    trips_start_hr = trips[trips['event_types'] == 'trip_start'].copy()
else:
    trips_start_hr = trips.copy()

trips_start_hr['datetime_hour'] = trips_start_hr['start_time'].dt.floor('H')
trips_start_hr['date'] = trips_start_hr['datetime_hour'].dt.date
trips_start_hr['hour'] = trips_start_hr['datetime_hour'].dt.hour

# Aggregate trips to hourly
trips_per_hour = trips_start_hr.groupby(['date', 'hour']).size().reset_index(name='trips_per_hour')

# Prepare weather data for merging
weather['datetime'] = pd.to_datetime(weather['datetime'], errors='coerce')
weather['date'] = weather['datetime'].dt.date
weather_hourly = weather.groupby(['date', 'hour']).agg({
    'temp_c': 'mean',
    'temp_max_c': 'max',
    'temp_min_c': 'min',
    'wind_direction_deg': 'mean',
    'wind_speed_ms': 'mean',
    'rain_mm': 'sum',  # Sum rain over the hour
    'weekend': 'first',
    'day_of_the_month': 'first'
}).reset_index()

# Merge with weather
base_df = pd.merge(trips_per_hour, weather_hourly, on=['date', 'hour'], how='left')

print(f"Base dataframe created: {base_df.shape}")
print(f"Columns: {base_df.columns.tolist()}")

# =================================================
# MODEL SPECIFICATIONS
# =================================================

# Optimal 12-segment specification (from 5.1 analysis)
optimal_12segment_spec = {
    "name": "Final 12-Segment Model - Optimal Solution",

    # Same data filters as your successful models
    "weekday_only": True,     # Sun–Thu (weekend==0)
    "no_rain_only": True,     # rain_mm == 0

    # 12-segment daytime categorization - maximum granularity while maintaining interpretability
    "hour_seg": {
        "hour_col": "hour",
        "out_col": "daytime_segment",
        "edges": [-0.1, 0, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24],  # 13 edges for 12 bins
        "labels": ["early_early", "early_morning", "morning", "peak_morning",
                   "mid_morning", "early_afternoon", "mid_afternoon", "late_afternoon",
                   "early_evening", "evening", "late_evening", "midnight"]  # 12 labels for 13 edges
    },

    # Model terms (same as your working models)
    "terms": [
        "C(daytime_segment)",  # 12-segment fixed effects
        "temp_c",              # Temperature
        "wind_speed_ms"        # Wind speed
    ],

    # Technical settings
    "cov_type": "HC0",        # Robust standard errors
    "plots": False,           # Disable plots for script
    "axis_max": 25,           # Consistent axis scaling

    # Post-hoc analysis to validate segment homogeneity
    "posthoc": {
        "factor_col": "daytime_segment",
        "correction": "holm"   # Multiple testing correction
    }
}

# =================================================
# UTILITY FUNCTIONS
# =================================================

def create_time_segments(df, spec):
    """Create time segment column based on specification"""
    hour_col = spec["hour_seg"]["hour_col"]
    out_col = spec["hour_seg"]["out_col"]
    edges = spec["hour_seg"]["edges"]
    labels = spec["hour_seg"]["labels"]

    df[out_col] = pd.cut(df[hour_col], bins=edges, labels=labels, include_lowest=True)
    df[out_col] = df[out_col].astype("category")
    return df

def filter_data(df, spec):
    """Apply data filters based on specification"""
    filtered_df = df.copy()

    if spec.get("weekday_only", False):
        if 'weekend' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['weekend'] == 0]
        elif 'day_of_week' in filtered_df.columns:
            # Assuming day_of_week: 0=Monday, 6=Sunday
            filtered_df = filtered_df[filtered_df['day_of_week'] < 5]  # Mon-Thu

    if spec.get("no_rain_only", False):
        filtered_df = filtered_df[filtered_df['rain_mm'] == 0]

    return filtered_df

def run_poisson_model(base_df, spec):
    """Run Poisson regression model with post-hoc analysis"""

    # Create working dataframe
    df = base_df.copy()

    # Apply filters
    df = filter_data(df, spec)

    # Create time segments
    if "hour_seg" in spec:
        df = create_time_segments(df, spec)

    # Create derived columns
    if "derived_cols" in spec:
        for col_name, expression in spec["derived_cols"].items():
            if isinstance(expression, str) and expression.startswith("pd.cut"):
                # Handle pd.cut expressions by evaluating in local scope
                local_vars = {'pd': pd, 'df': df, 'rain_mm': df['rain_mm'], 'np': np}
                try:
                    df[col_name] = eval(expression, {"__builtins__": {}}, local_vars)
                except:
                    # Fallback: try to execute the expression directly
                    exec(f"df['{col_name}'] = {expression}", {'pd': pd, 'df': df, 'rain_mm': df['rain_mm'], 'np': np, 'float': float, 'inf': float('inf')})
            else:
                # Evaluate expression in the context of the dataframe
                df[col_name] = df.eval(expression)

    # Prepare model formula
    formula = " + ".join(spec["terms"])
    formula = f"trips_per_hour ~ {formula}"

    print(f"Running model: {spec['name']}")
    print(f"Formula: {formula}")
    print(f"Data shape: {df.shape}")

    # Fit model
    try:
        model = sm.formula.glm(formula=formula, data=df, family=sm.families.Poisson(),
                              cov_type=spec.get("cov_type", "HC0")).fit()

        # Calculate dispersion
        y_pred = model.predict(df)
        residuals = df['trips_per_hour'] - y_pred
        dispersion = np.sum(residuals**2) / (len(df) - len(model.params))

        print(f"Model fitted successfully. Dispersion: {dispersion:.3f}")

        # Post-hoc analysis if specified
        posthoc_results = None
        if "posthoc" in spec:
            factor_col = spec["posthoc"]["factor_col"]
            if factor_col in df.columns:
                try:
                    posthoc_results = pairwise_tukeyhsd(
                        endog=df['trips_per_hour'],
                        groups=df[factor_col],
                        alpha=0.05
                    )
                    print(f"Post-hoc analysis completed for {factor_col}")
                except Exception as e:
                    print(f"Post-hoc analysis failed: {e}")

        return {
            'model': model,
            'dispersion': dispersion,
            'data': df,
            'posthoc': posthoc_results,
            'spec': spec
        }

    except Exception as e:
        print(f"Model fitting failed: {e}")
        return None

# =================================================
# MODEL 3A: 12-Segment + Continuous Rain Variable
# =================================================

print("\n" + "=" * 70)
print("MODEL 3A: 12-Segment + Continuous Rain Variable")
print("=" * 70)

model_3a_spec = optimal_12segment_spec.copy()
model_3a_spec["name"] = "Model 3A: 12-Segment + Continuous Rain"
model_3a_spec["no_rain_only"] = False  # Include ALL days (rain and no-rain)
model_3a_spec["terms"] = [
    "C(daytime_segment)",  # 12-segment fixed effects
    "temp_c",              # Temperature
    "wind_speed_ms",       # Wind speed
    "rain_mm"              # CONTINUOUS rain variable
]

out_3a = run_poisson_model(base_df, model_3a_spec)

if out_3a:
    print("\nModel 3A Results:")
    print(f"Dispersion: {out_3a['dispersion']:.3f}")
    if 'rain_mm' in out_3a['model'].params:
        print(".4f")
        print(".4f")
    else:
        print("Rain coefficient not found in model")
else:
    print("Model 3A failed to run")

# =================================================
# MODEL 3B: 12-Segment + Binary Rain Indicator
# =================================================

print("\n" + "=" * 70)
print("MODEL 3B: 12-Segment + Binary Rain Indicator")
print("=" * 70)

model_3b_spec = optimal_12segment_spec.copy()
model_3b_spec["name"] = "Model 3B: 12-Segment + Binary Rain"
model_3b_spec["no_rain_only"] = False  # Include ALL days

# Add binary rain indicator
model_3b_spec["derived_cols"] = {
    "rain_binary": "rain_mm > 0"
}

model_3b_spec["terms"] = [
    "C(daytime_segment)",  # 12-segment fixed effects
    "temp_c",              # Temperature
    "wind_speed_ms",       # Wind speed
    "C(rain_binary)"       # Binary rain indicator
]

out_3b = run_poisson_model(base_df, model_3b_spec)

if out_3b:
    print("\nModel 3B Results:")
    print(f"Dispersion: {out_3b['dispersion']:.3f}")
    binary_coeff_name = 'C(rain_binary)[T.True]'
    if binary_coeff_name in out_3b['model'].params:
        print(".4f")
        print(".4f")
    else:
        print("Binary rain coefficient not found in model")
else:
    print("Model 3B failed to run")

# =================================================
# MODEL 3C: 12-Segment + Rain Segmentation Analysis
# =================================================

print("\n" + "=" * 70)
print("MODEL 3C: 12-Segment + Rain Segmentation Analysis")
print("=" * 70)

# First, let's examine rain distribution to determine good segmentation
rain_stats = base_df['rain_mm'].describe()
print("Rain distribution analysis:")
print(rain_stats)

# Create different rain segments for post-hoc analysis
model_3c_spec = optimal_12segment_spec.copy()
model_3c_spec["name"] = "Model 3C: 12-Segment + Rain Segmentation"
model_3c_spec["no_rain_only"] = False  # Include ALL days

# Define rain segments based on distribution analysis
model_3c_spec["derived_cols"] = {
    "rain_segment": "pd.cut(rain_mm, bins=[-0.1, 0.1, 5, 15, float('inf')], labels=['no_rain', 'light_rain', 'moderate_rain', 'heavy_rain'])"
}

model_3c_spec["terms"] = [
    "C(daytime_segment)",  # 12-segment fixed effects
    "C(rain_segment)",     # Rain segment fixed effects
    "temp_c",              # Temperature
    "wind_speed_ms"        # Wind speed
]

# Add post-hoc analysis for rain segments
model_3c_spec["posthoc"] = {
    "factor_col": "rain_segment",
    "correction": "holm"
}

out_3c = run_poisson_model(base_df, model_3c_spec)

if out_3c:
    print("\nModel 3C Results:")
    print(f"Dispersion: {out_3c['dispersion']:.3f}")

    # Analyze rain segment effects
    rain_coeffs = out_3c['model'].params.filter(like='rain_segment')
    rain_pvals = out_3c['model'].pvalues.filter(like='rain_segment')

    print("\nRain Segment Coefficients:")
    for segment in ['light_rain', 'moderate_rain', 'heavy_rain']:
        coeff_name = f'C(rain_segment)[T.{segment}]'
        if coeff_name in rain_coeffs.index:
            print(".4f")
else:
    print("Model 3C failed to run")

# =================================================
# RAIN SEGMENTATION POST-HOC ANALYSIS
# =================================================

def perform_rain_segmentation_posthoc(base_df, model_output):
    """Perform post-hoc analysis on rain segments similar to time segments"""

    if not model_output or 'data' not in model_output:
        print("No valid model output for post-hoc analysis")
        return None

    df = model_output['data']

    if 'rain_segment' not in df.columns:
        print("rain_segment column not found in data")
        return None

    # Get the rain segment predictions
    rain_segments = ['no_rain', 'light_rain', 'moderate_rain', 'heavy_rain']

    # Calculate mean trips per rain segment
    rain_segment_means = {}
    for segment in rain_segments:
        mask = df['rain_segment'] == segment
        if mask.sum() > 0:
            rain_segment_means[segment] = df.loc[mask, 'trips_per_hour'].mean()

    print("\nRain Segment Analysis:")
    print("=" * 40)
    for segment, mean_trips in rain_segment_means.items():
        print(".2f")

    # Perform pairwise comparisons (like you did for time segments)
    # Filter to segments with enough data
    valid_segments = [s for s in rain_segments if (df['rain_segment'] == s).sum() >= 10]

    if len(valid_segments) >= 2:
        tukey_data = df[df['rain_segment'].isin(valid_segments)].copy()

        tukey_result = pairwise_tukeyhsd(
            endog=tukey_data['trips_per_hour'],
            groups=tukey_data['rain_segment'],
            alpha=0.05
        )

        print("\nPost-hoc Tukey HSD test for rain segments:")
        print(tukey_result)

        # Count significant differences
        sig_diffs = tukey_result.reject.sum()
        print(f"\nSignificant differences found: {sig_diffs} out of {len(tukey_result.reject)} comparisons")

    return rain_segment_means

print("\n" + "=" * 70)
print("RAIN SEGMENTATION POST-HOC ANALYSIS")
print("=" * 70)

rain_segment_analysis = perform_rain_segmentation_posthoc(base_df, out_3c)

# =================================================
# OPTIMIZE RAIN SEGMENTATION
# =================================================

def optimize_rain_segmentation(base_df, time_segments=12):
    """Test different rain segmentations to find optimal granularity"""

    # Different rain segmentation schemes to test
    rain_segmentation_schemes = [
        {
            'name': '2-segment',
            'bins': [-0.1, 0.1, float('inf')],
            'labels': ['no_rain', 'any_rain']
        },
        {
            'name': '3-segment',
            'bins': [-0.1, 0.1, 5, float('inf')],
            'labels': ['no_rain', 'light_rain', 'moderate_plus']
        },
        {
            'name': '4-segment',
            'bins': [-0.1, 0.1, 2, 10, float('inf')],
            'labels': ['no_rain', 'very_light', 'moderate', 'heavy']
        },
        {
            'name': '5-segment',
            'bins': [-0.1, 0.1, 1, 5, 15, float('inf')],
            'labels': ['no_rain', 'trace', 'light', 'moderate', 'heavy']
        }
    ]

    results = []

    for scheme in rain_segmentation_schemes:
        print(f"\nTesting {scheme['name']} rain segmentation...")

        # Create temporary dataframe with this segmentation
        temp_df = base_df.copy()
        temp_df['rain_segment_test'] = pd.cut(
            temp_df['rain_mm'],
            bins=scheme['bins'],
            labels=scheme['labels']
        )

        # Build model specification
        test_spec = optimal_12segment_spec.copy()
        test_spec["name"] = f"Test {scheme['name']} Rain Segmentation"
        test_spec["no_rain_only"] = False
        test_spec["derived_cols"] = {"rain_segment": f"pd.cut(rain_mm, bins={scheme['bins']}, labels={scheme['labels']})"}
        test_spec["terms"] = [
            "C(daytime_segment)",  # Keep the optimal 12 time segments
            "C(rain_segment)",
            "temp_c",
            "wind_speed_ms"
        ]

        # Run model
        try:
            test_output = run_poisson_model(temp_df, test_spec)
            if test_output:
                dispersion = test_output['dispersion']

                # Count significant differences within rain segments
                if test_output.get('posthoc') is not None:
                    sig_diffs = len(test_output['posthoc'].reject)
                else:
                    sig_diffs = 0

                results.append({
                    'scheme': scheme['name'],
                    'dispersion': dispersion,
                    'sig_diffs': sig_diffs,
                    'segments': len(scheme['labels'])
                })

                print(".3f")
            else:
                results.append({
                    'scheme': scheme['name'],
                    'dispersion': float('inf'),
                    'sig_diffs': 0,
                    'segments': len(scheme['labels'])
                })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'scheme': scheme['name'],
                'dispersion': float('inf'),
                'sig_diffs': 0,
                'segments': len(scheme['labels'])
            })

    # Show results
    results_df = pd.DataFrame(results)
    print("\nRain Segmentation Optimization Results:")
    print("=" * 50)
    print(results_df.to_string(index=False))

    # Find optimal segmentation
    if len(results_df) > 0 and not results_df['dispersion'].empty:
        optimal = results_df.loc[results_df['dispersion'].idxmin()]
        print(f"\nOptimal rain segmentation: {optimal['scheme']}")
        print(".3f")
    else:
        optimal = None
        print("\nNo valid results found")

    return results_df, optimal

print("\n" + "=" * 70)
print("OPTIMIZE RAIN SEGMENTATION")
print("=" * 70)

rain_opt_results, optimal_rain_seg = optimize_rain_segmentation(base_df)

# =================================================
# FINAL MODEL: Optimal Time + Rain Segmentation
# =================================================

print("\n" + "=" * 70)
print("FINAL MODEL: Optimal Time + Rain Segmentation")
print("=" * 70)

# Use the optimal rain segmentation from above
if optimal_rain_seg is not None:
    # Get the optimal scheme details
    optimal_scheme_name = optimal_rain_seg['scheme']

    # Map scheme names to bins/labels
    scheme_configs = {
        '2-segment': ([-0.1, 0.1, float('inf')], ['no_rain', 'any_rain']),
        '3-segment': ([-0.1, 0.1, 5, float('inf')], ['no_rain', 'light_rain', 'moderate_plus']),
        '4-segment': ([-0.1, 0.1, 2, 10, float('inf')], ['no_rain', 'very_light', 'moderate', 'heavy']),
        '5-segment': ([-0.1, 0.1, 1, 5, 15, float('inf')], ['no_rain', 'trace', 'light', 'moderate', 'heavy'])
    }

    if optimal_scheme_name in scheme_configs:
        final_rain_bins, final_rain_labels = scheme_configs[optimal_scheme_name]
    else:
        # Default fallback
        final_rain_bins = [-0.1, 0.1, 5, 15, float('inf')]
        final_rain_labels = ['no_rain', 'light_rain', 'moderate_rain', 'heavy_rain']
else:
    # Default fallback
    final_rain_bins = [-0.1, 0.1, 5, 15, float('inf')]
    final_rain_labels = ['no_rain', 'light_rain', 'moderate_rain', 'heavy_rain']

final_model_spec = {
    "name": "Final Model: 12-Time Segments + Optimal Rain Segmentation",

    "weekday_only": True,     # Sun–Thu only
    "no_rain_only": False,    # Include all weather

    # Optimal 12 time segments
    "hour_seg": {
        "hour_col": "hour",
        "out_col": "daytime_segment",
        "edges": [-0.1, 0, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 24],
        "labels": ["early_early", "early_morning", "morning", "peak_morning",
                   "mid_morning", "early_afternoon", "mid_afternoon", "late_afternoon",
                   "early_evening", "evening", "late_evening", "midnight"]
    },

    # Optimal rain segmentation
    "derived_cols": {
        "rain_segment": f"pd.cut(rain_mm, bins={final_rain_bins}, labels={final_rain_labels})"
    },

    "terms": [
        "C(daytime_segment)",  # 12 time segments
        "C(rain_segment)",     # Optimal rain segments
        "temp_c",              # Temperature
        "wind_speed_ms"        # Wind speed
    ],

    "cov_type": "HC0",
    "plots": False,

    # Post-hoc for both factors
    "posthoc": {
        "factor_col": "rain_segment",  # Focus on rain segmentation
        "correction": "holm"
    }
}

final_output = run_poisson_model(base_df, final_model_spec)

if final_output:
    print("\nFinal Model Results:")
    print(f"Dispersion: {final_output['dispersion']:.3f}")
    print(".3f")

    # Compare with baseline (no rain model)
    baseline_dispersion = 2.096  # From your 5.1 analysis
    improvement = ((baseline_dispersion - final_output['dispersion']) / baseline_dispersion) * 100
    print(".1f")
else:
    print("Final model failed to run")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("Summary of key findings:")
if out_3a and 'model' in out_3a:
    print(".3f")
if out_3b and 'model' in out_3b:
    print(".3f")
if out_3c and 'model' in out_3c:
    print(".3f")
if final_output:
    print(".3f")