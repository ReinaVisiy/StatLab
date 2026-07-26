"""
Correlation Matrix & Principal Component Spectrum Module.
Exports: run_correlation_matrix
"""
import numpy as np
import pandas as pd
from i18n.translations import t as tt

def run_correlation_matrix(data_frame: pd.DataFrame, var_cols: list = None, lang: str = "en") -> dict:
    if var_cols is None or len(var_cols) < 2:
        # Select all numeric columns
        df_num = data_frame.select_dtypes(include=[np.number])
    else:
        df_num = data_frame[var_cols].select_dtypes(include=[np.number])

    if df_num.shape[1] < 2:
        raise ValueError("At least 2 numeric variables are required to compute a correlation matrix.")

    corr_df = df_num.corr(method="pearson")
    var_names = list(corr_df.columns)

    # Eigenvalue decomposition for Principal Component Analysis / Spectrum
    corr_matrix = corr_df.to_numpy()
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)

    # Sort descending
    sort_idx = np.argsort(eigenvals)[::-1]
    sorted_eigenvals = eigenvals[sort_idx]
    sorted_eigenvecs = eigenvecs[:, sort_idx]

    total_var = np.sum(sorted_eigenvals)
    var_explained = sorted_eigenvals / total_var
    cum_var_explained = np.cumsum(var_explained)

    pca_spectrum = []
    for i in range(len(sorted_eigenvals)):
        pca_spectrum.append({
            "PC": f"PC{i+1}",
            "Eigenvalue": float(sorted_eigenvals[i]),
            "Variance Explained (%)": float(var_explained[i] * 100.0),
            "Cumulative Variance (%)": float(cum_var_explained[i] * 100.0)
        })

    steps = [
        tt("correlation_matrix_computed", lang).format(rows=corr_df.shape[0], cols=corr_df.shape[1]),
        f"{tt('eigen_decomposition_label', lang)} = {total_var:.2f}",
        f"PC1 {tt('pc_explains_label', lang)}: {var_explained[0]*100:.2f}%",
        f"PC1+PC2 {tt('cumulative_variance_label', lang)} 2 PCs: {cum_var_explained[min(1, len(cum_var_explained)-1)]*100:.2f}%"
    ]

    return {
        "var_names": var_names,
        "correlation_matrix": corr_df.astype(float).to_dict(),
        "pca_spectrum": pca_spectrum,
        "eigenvalues": sorted_eigenvals.tolist(),
        "eigenvectors": sorted_eigenvecs.tolist(),
        "heatmap_data": {
            "z": corr_matrix.tolist(),
            "x": var_names,
            "y": var_names
        },
        "steps": steps
    }
