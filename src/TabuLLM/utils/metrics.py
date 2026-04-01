import pandas as pd
import numpy as np
from simsimd import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

def compute_jensenshannon_fidelity(a: pd.Series, b: pd.Series, bins: int = 20) -> float:
    """
    Computes the Jensen-Shannon Divergence (JSD) based fidelity between two numerical series.
    
    Args:
        a (pd.Series): The original numerical data series.
        b (pd.Series): The augmented or synthetic numerical data series.
        bins (int, optional): The number of bins to use for the histogram. Defaults to 20.
        
    Returns:
        float: A fidelity score between 0 and 1, where 1 indicates identical distributions.
               Returns 0 if the series are too small or JSD is NaN.
    """
    a_clean = a.dropna()
    b_clean = b.dropna()

    if len(a_clean) <= 5 or len(b_clean) <= 5:
        return 0.0

    hist_a, bin_edges = np.histogram(a_clean, bins=bins, density=True)
    hist_b, _ = np.histogram(b_clean, bins=bin_edges, density=True)
    
    jsd = jensenshannon(hist_a, hist_b)
    if np.isnan(jsd):
        return 0.0
        
    return float(1.0 - jsd)

def compute_z_drift(a: pd.Series, b: pd.Series) -> float:
    """
    Computes the Z-Drift (Mean Shift standardized by original variance) between two numerical series.
    
    Args:
        a (pd.Series): The original numerical data series.
        b (pd.Series): The augmented or synthetic numerical data series.
        
    Returns:
        float: The absolute Z-drift score.
    """
    a_clean = a.dropna()
    b_clean = b.dropna()
    
    if len(a_clean) == 0 or len(b_clean) == 0:
        return 0.0
        
    std_a = a_clean.std()
    # Prevent division by zero
    std_a = std_a if std_a > 0 else 1.0 
    
    return float(abs(a_clean.mean() - b_clean.mean()) / std_a)

def compute_mean_relative_error(a: pd.Series, b: pd.Series) -> float:
    """
    Computes the Mean Relative Error (MRE) of the mean values between two numerical series.
    
    Args:
        a (pd.Series): The original numerical data series.
        b (pd.Series): The augmented or synthetic numerical data series.
        
    Returns:
        float: The relative error of the mean. Returns 0 if original mean is 0.
    """
    a_clean = a.dropna()
    b_clean = b.dropna()
    
    if len(a_clean) == 0 or len(b_clean) == 0:
        return 0.0
        
    mean_a = a_clean.mean()
    if mean_a == 0:
        return 0.0
        
    return float(abs(mean_a - b_clean.mean()) / abs(mean_a))

def compute_correlation_stability(df_orig: pd.DataFrame, df_aug: pd.DataFrame, numeric_cols: list) -> float:
    """
    Computes the stability of the correlation matrix between the original and augmented datasets.
    
    Args:
        df_orig (pd.DataFrame): The original dataset.
        df_aug (pd.DataFrame): The augmented dataset.
        numeric_cols (list): List of numeric column names to compute correlations for.
        
    Returns:
        float: Correlation stability score scaled 0-100.
    """
    if len(numeric_cols) <= 1:
        return 0.0
        
    c_orig = df_orig[numeric_cols].corr().fillna(0).values
    c_aug = df_aug[numeric_cols].corr().fillna(0).values
    
    stability = 1.0 - np.mean(np.abs(c_orig - c_aug))
    return float(stability * 100.0)

def compute_privacy_metrics(df_orig: pd.DataFrame, df_aug: pd.DataFrame, numeric_cols: list) -> dict:
    """
    Computes privacy and distance-based structural metrics like Privacy Risk, Coverage Score, and NNDR.
    
    Privacy Risk: % of synthetic records dangerously close to real records.
    Coverage Score: % of synthetic records sufficiently far to provide diversity.
    Nearest Neighbor Distance Ratio (NNDR): Ratio of distance to closest real vs closest synthetic.
    
    Args:
        df_orig (pd.DataFrame): The original dataset.
        df_aug (pd.DataFrame): The augmented dataset.
        numeric_cols (list): List of numeric column names.
        
    Returns:
        dict: A dictionary containing 'privacy_risk', 'coverage_score', and 'nndr'.
    """
    results = {"privacy_risk": 0, "coverage_score": 0, "nndr": 0.0}
    
    if not numeric_cols or len(df_orig) == 0 or len(df_aug) == 0:
        return results

    scaler = MinMaxScaler()
    combined_numeric = pd.concat([df_orig[numeric_cols], df_aug[numeric_cols]]).fillna(0)
    scaler.fit(combined_numeric)

    orig_scaled = scaler.transform(df_orig[numeric_cols].fillna(0))
    aug_scaled = scaler.transform(df_aug[numeric_cols].fillna(0))

    sample_size = min(200, len(aug_scaled))
    aug_sample = aug_scaled[:sample_size]
    
    # Calculate pairwise distances (Rows: Augmented Sample, Cols: Original Full)
    dist_matrix = euclidean_distances(aug_sample, orig_scaled)
    # the closest original neighbor for each augmented sample
    min_distances = np.min(dist_matrix, axis=1)

    results["privacy_risk"] = int((min_distances < 0.05).mean() * 100)
    results["coverage_score"] = int((min_distances > 0.1).mean() * 100)

    if len(orig_scaled) > 1 and len(aug_sample) > 1:
        aug_to_aug = euclidean_distances(aug_sample, aug_sample)
        np.fill_diagonal(aug_to_aug, np.inf) # Ignore self-distance
        nn_aug = np.min(aug_to_aug, axis=1)
        nn_orig = min_distances
        results["nndr"] = float(np.mean(nn_orig / (nn_aug + 1e-6)))
        
    return results

def compute_hellinger_fidelity(a: pd.Series, b: pd.Series) -> float:
    """
    Computes the Hellinger Distance based fidelity between two categorical series.
    
    Args:
        a (pd.Series): The original categorical data series.
        b (pd.Series): The augmented or synthetic categorical data series.
        
    Returns:
        float: A fidelity score between 0 and 1, where 1 indicates identical proportion distributions.
    """
    p = a.value_counts(normalize=True).sort_index()
    q = b.value_counts(normalize=True).sort_index()
    
    all_cats = p.index.union(q.index)
    p = p.reindex(all_cats, fill_value=0).values
    q = q.reindex(all_cats, fill_value=0).values
    
    h_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
    return float(1.0 - h_dist)

def compute_entropy_preservation(a: pd.Series, b: pd.Series) -> float:
    """
    Computes how well the Shannon entropy is preserved between two numerical series.
    """
    def shannon_entropy(s):
        counts = s.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts + 1e-12))
    
    e_a = shannon_entropy(a.dropna())
    e_b = shannon_entropy(b.dropna())
    
    if e_a == 0: return 1.0 if e_b == 0 else 0.0
    return float(1.0 - abs(e_a - e_b) / e_a)

def compute_completeness(df: pd.DataFrame) -> float:
    """
    Computes the completeness (non-null percentage) of a dataset.
    
    Args:
        df (pd.DataFrame): The dataset to evaluate.
        
    Returns:
        float: Completeness score scaled 0-100.
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0
    
    null_count = df.isna().sum().sum()
    return float((total_cells - null_count) / total_cells * 100.0)

def calculate_metrics(df_orig: pd.DataFrame, df_aug: pd.DataFrame) -> dict:
    """
    Calculates a comprehensive suite of statistical similarity and structural metrics
    comparing an augmented/synthetic dataset to an original dataset.
    
    Args:
        df_orig (pd.DataFrame): The original source dataset.
        df_aug (pd.DataFrame): The newly generated or augmented dataset.
        
    Returns:
        dict: A dictionary of computed metrics including:
            - fidelity_score: (int) 0-100 numerical JSD fidelity average.
            - corr_stability: (int) 0-100 similarity of numerical feature correlation matrices.
            - distribution_consistency: (int) 0-100 median numerical JSD fidelity.
            - avg_z_drift: (float) Average Z-drift across numerical columns.
            - cat_fidelity: (int) 0-100 categorical Hellinger fidelity average.
            - privacy_risk: (int) 0-100 percentage of too-close near neighbors.
            - coverage_score: (int) 0-100 diversity coverage representing synthetic novelty.
            - completeness: (float) 0-100 percentage of non-null cells in augmented data.
            - mean_feature_error: (int) 0-100 inverse average mean relative error.
            - nndr: (float) Nearest Neighbor Distance Ratio.
    """
    # Strictly exclude internal metadata columns and only use columns present in both datasets
    internal_cols = ['_fold_id', '_row_id', 'is_context', 'is_target', 'ai_audit']
    common_cols = [c for c in df_aug.columns if c in df_orig.columns 
                   and not c.startswith('is_ai_') 
                   and c not in internal_cols]

    numeric_cols = [c for c in df_aug[common_cols].select_dtypes(include='number').columns]
    cat_cols = [c for c in common_cols if c not in numeric_cols]

    jsd_scores = []
    mre_list = []
    z_drift_list = []
    entropy_scores = []
    
    # Evaluate Numerical Columns Extensively
    if numeric_cols:
        for col in numeric_cols:
            a = df_orig[col]
            b = df_aug[col]

            jsd = compute_jensenshannon_fidelity(a, b)
            if jsd > 0:
                jsd_scores.append(jsd)
            
            z_drift_list.append(compute_z_drift(a, b))
            entropy_scores.append(compute_entropy_preservation(a, b))
            
            mre = compute_mean_relative_error(a, b)
            if mre > 0:
                mre_list.append(mre)

    # Prepare Encoded Data for All Features (Numeric + Categorical)
    # We use pd.factorize for categoricals to make them numeric for structural metrics
    df_orig_enc = df_orig.copy()
    df_aug_enc = df_aug.copy()
    
    all_features = numeric_cols + cat_cols
    
    for col in cat_cols:
        # Get combined unique values for consistent encoding
        combined = pd.concat([df_orig[col], df_aug[col]])
        codes, uniques = pd.factorize(combined)
        
        # Split back
        df_orig_enc[col] = codes[:len(df_orig)]
        df_aug_enc[col] = codes[len(df_orig):]

    # Calculate Aggregate Structural Metrics on ALL encoded features
    corr_stability = compute_correlation_stability(df_orig_enc, df_aug_enc, all_features)
    privacy_results = compute_privacy_metrics(df_orig_enc, df_aug_enc, all_features)
    
    # Evaluate Categorical Columns
    hellinger_cats = []
    if cat_cols:
        for col in cat_cols:
            hellinger_cats.append(compute_hellinger_fidelity(df_orig[col], df_aug[col]))

    # Compile Final Metric Dictionary
    return {
        "Synthetic Fidelity (JSD)": int(np.mean(jsd_scores) * 100) if jsd_scores else 0,
        "Correlation Stability": int(corr_stability),
        "Distribution Consistency": int(np.median(jsd_scores) * 100) if jsd_scores else 0,
        "Entropy Preservation": int(np.mean(entropy_scores) * 100) if entropy_scores else 0,
        "Mean Z Score Drift": float(np.mean(z_drift_list)) if z_drift_list else 0.0,
        "Categorical Fidelity": int(np.mean(hellinger_cats) * 100) if hellinger_cats else 0,
        "Privacy Risk": privacy_results["privacy_risk"],
        "Data Density": compute_completeness(df_aug),
        "Coverage Score": privacy_results["coverage_score"],
        "Feature Accuracy": int((1 - np.mean(mre_list)) * 100) if mre_list else 0,
        "Novelty Score": privacy_results["nndr"]
    }
