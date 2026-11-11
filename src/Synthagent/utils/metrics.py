import pandas as pd
import numpy as np
from simsimd import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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
    
    # Detect complete range mismatch
    if a_clean.max() < b_clean.min() or b_clean.max() < a_clean.min():
        return 0.0

    jsd = jensenshannon(hist_a, hist_b)
    
    if np.isnan(jsd) or (np.sum(hist_b) == 0 and np.sum(hist_a) > 0):
        return 0.0
        
    return float(max(0.0, 1.0 - jsd))

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
    Nearest Neighbor Distance Ratio (NNDR): the standard privacy-literature definition -- for
    each synthetic record, the ratio of its distance to the NEAREST real record over its
    distance to the SECOND-nearest real record (both distances measured against the real
    data, not synthetic-vs-synthetic). Bounded in [0, 1] by construction since the 1st-NN
    distance can never exceed the 2nd-NN distance: near 0 means the synthetic record sits
    suspiciously close to one specific real record relative to any other (memorization /
    re-identification risk); near 1 means it's roughly equidistant from multiple real
    records (safer, more novel).
    
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

    if len(orig_scaled) >= 2:
        # 1st- and 2nd-nearest REAL neighbor distance for every synthetic sample -- reusing
        # dist_matrix (aug_sample x orig_scaled) already computed above. This is naturally
        # bounded in [0, 1] (d1 <= d2 always), unlike comparing distance-to-real against
        # distance-to-nearest-OTHER-SYNTHETIC: that formulation blows up unboundedly whenever
        # two synthetic rows happen to be near-duplicates (a near-zero denominator), which is
        # exactly what was producing Novelty Score values in the thousands.
        sorted_dists = np.sort(dist_matrix, axis=1)
        d1, d2 = sorted_dists[:, 0], sorted_dists[:, 1]
        results["nndr"] = float(np.mean(d1 / (d2 + 1e-9)))

    return results

def compute_hellinger_fidelity(a: pd.Series, b: pd.Series) -> float:
    """
    Computes the Hellinger Distance based fidelity between two categorical series.
    """
    # Cast to string to prevent mixed-type comparison errors during sort_index
    p = a.astype(str).value_counts(normalize=True).sort_index()
    q = b.astype(str).value_counts(normalize=True).sort_index()
    
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
        # Cast to string for consistent categorical entropy calculation
        counts = s.astype(str).value_counts(normalize=True)
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
            - nndr: (float) Nearest Neighbor Distance Ratio, in [0, 1] -- see
              compute_privacy_metrics() docstring; near 1 = safer/more novel.
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

def calculate_metrics_by_class(df_orig: pd.DataFrame, df_aug: pd.DataFrame, target_col: str) -> dict:
    """
    Computes calculate_metrics() separately for the minority and majority classes of
    target_col (by prevalence in df_orig), isolating whether synthetic fidelity holds
    up on the minority class specifically -- the class an augmentation pipeline is
    actually meant to help on an imbalanced dataset.

    Args:
        df_orig (pd.DataFrame): The original source dataset.
        df_aug (pd.DataFrame): The newly generated or augmented dataset.
        target_col (str): The classification target column to group by.

    Returns:
        dict: {"minority_label": ..., "majority_label": ...,
               "minority": {<calculate_metrics() output>}, "majority": {...}}
              Missing "minority"/"majority" keys (or an empty dict) when target_col is
              absent, has fewer than 2 classes, or a class subset is too small (<6 rows)
              for the underlying fidelity metrics to be meaningful.
    """
    if target_col not in df_orig.columns or target_col not in df_aug.columns:
        return {}

    counts = df_orig[target_col].value_counts()
    if len(counts) < 2:
        return {}

    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    out: dict = {"minority_label": str(minority_label), "majority_label": str(majority_label)}
    for role, label in (("minority", minority_label), ("majority", majority_label)):
        orig_sub = df_orig[df_orig[target_col] == label]
        aug_sub = df_aug[df_aug[target_col] == label]
        if len(orig_sub) < 6 or len(aug_sub) < 6:
            continue
        out[role] = calculate_metrics(orig_sub, aug_sub)
    return out


def _encode_common_columns(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: list) -> tuple:
    """
    Factorizes categorical columns consistently across both frames (using their combined
    unique values, mirroring the encoding already used in calculate_metrics) and returns
    (df_a_encoded, df_b_encoded) restricted to `cols`, both numeric and NaN-free.
    """
    a_enc = df_a[cols].copy()
    b_enc = df_b[cols].copy()
    for col in cols:
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            continue
        combined = pd.concat([df_a[col], df_b[col]])
        codes, _ = pd.factorize(combined)
        a_enc[col] = codes[:len(df_a)]
        b_enc[col] = codes[len(df_a):]
    return a_enc.fillna(0), b_enc.fillna(0)


def compute_c2st_accuracy(df_real: pd.DataFrame, df_synth: pd.DataFrame, sample_cap: int = 500):
    """
    Classifier Two-Sample Test: trains a RandomForest to distinguish real (label 0) from
    synthetic (label 1) rows on a held-out split, and returns its test accuracy.
    ~0.5 = indistinguishable (high fidelity); higher = the classifier can tell them apart
    (lower fidelity). Returns None if there are too few rows for a meaningful test.
    """
    if len(df_real) < 10 or len(df_synth) < 10:
        return None

    common_cols = [c for c in df_synth.columns if c in df_real.columns]
    if not common_cols:
        return None

    real_sample = df_real[common_cols].sample(min(len(df_real), sample_cap), random_state=42)
    synth_sample = df_synth[common_cols].sample(min(len(df_synth), sample_cap), random_state=42)

    real_enc, synth_enc = _encode_common_columns(real_sample, synth_sample, common_cols)
    X = pd.concat([real_enc, synth_enc], ignore_index=True)
    y = np.array([0] * len(real_enc) + [1] * len(synth_enc))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return None
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(X_train, y_train)
        return float(accuracy_score(y_test, clf.predict(X_test)))
    except Exception:
        return None


def compute_mia_attack_auc(df_members: pd.DataFrame, df_nonmembers: pd.DataFrame, df_synth: pd.DataFrame):
    """
    Simplified distance-based membership-inference attack. For every real record --
    members (rows the pipeline used as train/context data) and non-members (the held-out
    test split it never saw) -- the attacker's score is the negative nearest-neighbour
    distance to the synthetic corpus: closer to synthetic data => guessed "member". Returns
    the attack's ROC-AUC (0.5 = no leakage signal; higher = a successful attack, meaning
    the synthetic data leaks information about which real rows were used to generate it).
    """
    if len(df_members) < 5 or len(df_nonmembers) < 5 or len(df_synth) < 5:
        return None

    common_cols = [c for c in df_synth.columns if c in df_members.columns and c in df_nonmembers.columns]
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(df_members[c])]
    if not numeric_cols:
        return None

    try:
        combined = pd.concat([
            df_members[numeric_cols], df_nonmembers[numeric_cols], df_synth[numeric_cols]
        ]).fillna(0)
        scaler = MinMaxScaler()
        scaler.fit(combined)

        members_scaled = scaler.transform(df_members[numeric_cols].fillna(0))
        nonmembers_scaled = scaler.transform(df_nonmembers[numeric_cols].fillna(0))
        synth_scaled = scaler.transform(df_synth[numeric_cols].fillna(0))

        member_dist = euclidean_distances(members_scaled, synth_scaled).min(axis=1)
        nonmember_dist = euclidean_distances(nonmembers_scaled, synth_scaled).min(axis=1)

        labels = np.array([1] * len(member_dist) + [0] * len(nonmember_dist))
        scores = np.concatenate([-member_dist, -nonmember_dist])

        if len(np.unique(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def compute_schema_validity(df_real: pd.DataFrame, df_gen: pd.DataFrame) -> "float | None":
    """
    Percentage of generated rows that are schema-valid against the real table's
    observed domain, judged identically for every generator:
      - every real column is present and non-empty;
      - numerical columns parse as numbers inside the observed [min, max];
      - columns whose observed values are all integers receive integers;
      - non-numerical columns take a value observed in the real table.
    Returns None when nothing was generated.
    """
    if df_gen is None or len(df_gen) == 0:
        return None
    valid = pd.Series(True, index=df_gen.index)
    for col in df_real.columns:
        if col not in df_gen.columns:
            return 0.0
        real = df_real[col].dropna()
        gen = df_gen[col]
        if pd.api.types.is_numeric_dtype(real):
            num = pd.to_numeric(gen, errors="coerce")
            ok = num.notna() & (num >= real.min()) & (num <= real.max())
            if np.allclose(real, np.round(real)):
                ok &= np.isclose(num, np.round(num))
        else:
            ok = gen.notna() & gen.astype(str).isin(set(real.astype(str)))
        valid &= ok.fillna(False).astype(bool)
    return 100.0 * float(valid.mean())
