import pandas as pd
import numpy as np
from simsimd import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

def calculate_metrics(df_orig, df_aug):
    numeric_cols = df_aug.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in df_aug.columns if c not in numeric_cols]

    # METRIC ACCUMULATORS
    jsd_scores = []
    mre_list = []
    hellinger_cats = []
    z_drift_list = []

    corr_stability = 0
    privacy_risk = 0
    coverage_score = 0
    nndr = 0

    if numeric_cols:
        for col in numeric_cols:
            a = df_orig[col].dropna()
            b = df_aug[col].dropna()

            if len(a) > 5 and len(b) > 5:
                hist_a, bin_edges = np.histogram(a, bins=20, density=True)
                hist_b, _ = np.histogram(b, bins=bin_edges, density=True)
                jsd = jensenshannon(hist_a, hist_b)
                jsd_scores.append(1 - jsd if not np.isnan(jsd) else 0)

                z_drift = abs(a.mean() - b.mean()) / (a.std() if a.std() > 0 else 1)
                z_drift_list.append(z_drift)

                if a.mean() != 0:
                    mre = abs(a.mean() - b.mean()) / abs(a.mean())
                    mre_list.append(mre)

        if len(numeric_cols) > 1:
            c_orig = df_orig[numeric_cols].corr().fillna(0).values
            c_aug = df_aug[numeric_cols].corr().fillna(0).values
            corr_stability = (1 - np.mean(np.abs(c_orig - c_aug))) * 100

        scaler = MinMaxScaler()
        combined_numeric = pd.concat([df_orig[numeric_cols], df_aug[numeric_cols]]).fillna(0)
        scaler.fit(combined_numeric)

        orig_scaled = scaler.transform(df_orig[numeric_cols].fillna(0))
        aug_scaled = scaler.transform(df_aug[numeric_cols].fillna(0))

        sample_size = min(200, len(aug_scaled))
        aug_sample = aug_scaled[:sample_size]
        dist_matrix = euclidean_distances(aug_sample, orig_scaled)
        min_distances = np.min(dist_matrix, axis=1)

        privacy_risk = int((min_distances < 0.05).mean() * 100)
        coverage_score = int((min_distances > 0.1).mean() * 100)

        if len(orig_scaled) > 1 and len(aug_sample) > 1:
            aug_to_aug = euclidean_distances(aug_sample, aug_sample)
            np.fill_diagonal(aug_to_aug, np.inf)
            nn_aug = np.min(aug_to_aug, axis=1)
            nn_orig = min_distances
            nndr = np.mean(nn_orig / (nn_aug + 1e-6))

    if cat_cols:
        for col in cat_cols:
            p = df_orig[col].value_counts(normalize=True).sort_index()
            q = df_aug[col].value_counts(normalize=True).sort_index()
            all_cats = p.index.union(q.index)
            p = p.reindex(all_cats, fill_value=0).values
            q = q.reindex(all_cats, fill_value=0).values
            h_dist = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
            hellinger_cats.append(1 - h_dist)

    fidelity_score = int(np.mean(jsd_scores) * 100) if jsd_scores else 0
    avg_z_drift = np.mean(z_drift_list) if z_drift_list else 0
    mean_feature_error = int((1 - np.mean(mre_list)) * 100) if mre_list else 0
    cat_fidelity = int(np.mean(hellinger_cats) * 100) if hellinger_cats else 0
    distribution_consistency = int(np.median(jsd_scores) * 100) if jsd_scores else 0

    orig_nulls = df_orig.isna().sum().sum()
    aug_nulls = df_aug.isna().sum().sum()
    total_cells_aug = df_aug.shape[0] * df_aug.shape[1]
    completeness = ((total_cells_aug - aug_nulls) / total_cells_aug * 100) if total_cells_aug > 0 else 0

    return {
        "fidelity_score": fidelity_score,
        "corr_stability": corr_stability,
        "distribution_consistency": distribution_consistency,
        "avg_z_drift": avg_z_drift,
        "cat_fidelity": cat_fidelity,
        "privacy_risk": privacy_risk,
        "coverage_score": coverage_score,
        "completeness": completeness,
        "mean_feature_error": mean_feature_error,
        "nndr": nndr
    }
