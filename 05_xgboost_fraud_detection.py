import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <!--
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook 05: XGBoost Fraud Detection with Transaction Foundation Model Embeddings

    We compare three XGBoost configurations with HPO-optimized hyperparameters to evaluate how foundation model embeddings improve fraud detection:

    | Model | Features | Description |
    |-------|----------|-------------|
    | **Baseline** | 13d raw tabular features | All available raw fields |
    | **Embeddings** | 64d PCA embeddings | Foundation model embeddings only |
    | **Combined** | 13d raw + 64d PCA embeddings | Raw features augmented with foundation model embeddings |

    | Split | Samples | Fraud Rate | Purpose |
    |-------|---------|------------|---------|
    | **Train** | 1M (balanced) | \~2.5% | XGBoost training |
    | **Val** | 100K (stratified) | \~0.1% | Early stopping |
    | **Test** | 100K (stratified) | \~0.1% | Final holdout evaluation |

    - **Balanced training**: All fraud + sampled normal transactions from the full training split
    - **Consistent evaluation**: Val/Test use the same 100K stratified subsets created in notebook 01

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Setup and Imports
    """)
    return


@app.cell
def _():
    import sys
    import json
    import time
    import warnings
    warnings.filterwarnings('ignore')

    from pathlib import Path

    import numpy as np
    import cudf
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.decomposition import PCA

    import xgboost as xgb
    import torch

    print(f"cuDF version:    {cudf.__version__}")
    print(f"XGBoost version: {xgb.__version__}")
    print(f"GPU available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU 0:           {props.name}, {props.total_memory / 1e9:.1f} GB")

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    XGB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    PROJECT_ROOT = Path(".").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    EMBED_DIR = PROJECT_ROOT / "data" / "embeddings"
    TEMPORAL_DIR = PROJECT_ROOT / "data" / "TabFormer" / "temporal_split"
    OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    assert EMBED_DIR.exists(), f"Embeddings directory not found: {EMBED_DIR}"
    assert TEMPORAL_DIR.exists(), f"Temporal split directory not found: {TEMPORAL_DIR}"

    print(f"\nEmbedding directory: {EMBED_DIR}")
    print(f"Temporal split data: {TEMPORAL_DIR}")
    print(f"Outputs directory:   {OUTPUTS_DIR}")
    return (
        EMBED_DIR,
        OUTPUTS_DIR,
        OrdinalEncoder,
        PCA,
        TEMPORAL_DIR,
        XGB_DEVICE,
        average_precision_score,
        cudf,
        json,
        make_column_selector,
        make_column_transformer,
        np,
        plt,
        roc_auc_score,
        time,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 1. Load Embeddings and Metadata

    Load pre-extracted transaction foundation model embeddings (last-token pooling). Training embeddings correspond to a balanced 1M sample from the full training split, while val/test embeddings match the 100k stratified subsets from notebook 01.
    """)
    return


@app.cell
def _(EMBED_DIR, json, np, time):
    with open(EMBED_DIR / 'metadata.json') as f:
        metadata = json.load(f)
    print('Embedding metadata:')
    for key, value in metadata.items():
        if not isinstance(value, dict):
            print(f'  {key}: {value}')
    print('\nLoading embeddings...')
    _t0 = time.time()
    # ------------------------------------------------------------------
    # Load pre-computed train / val / test embeddings (foundation model, last-token)
    X_train_embed_raw = np.load(EMBED_DIR / 'train_embeddings.npy')
    y_train = np.load(EMBED_DIR / 'train_labels.npy')
    train_row_ids = np.load(EMBED_DIR / 'train_row_ids.npy')
    X_val_embed_raw = np.load(EMBED_DIR / 'val_embeddings.npy')
    y_val = np.load(EMBED_DIR / 'val_labels.npy')
    val_row_ids = np.load(EMBED_DIR / 'val_row_ids.npy')
    X_test_embed_raw = np.load(EMBED_DIR / 'test_embeddings.npy')
    y_test = np.load(EMBED_DIR / 'test_labels.npy')
    test_row_ids = np.load(EMBED_DIR / 'test_row_ids.npy')
    _load_time = time.time() - _t0
    print(f'Loaded in {_load_time:.1f}s')
    n_train = len(X_train_embed_raw)
    n_val = len(X_val_embed_raw)
    n_test = len(X_test_embed_raw)
    assert X_train_embed_raw.shape[0] == metadata['n_train'], 'Train size mismatch'
    assert X_val_embed_raw.shape[0] == metadata['n_val'], 'Val size mismatch'
    assert X_test_embed_raw.shape[0] == metadata['n_test'], 'Test size mismatch'
    assert X_train_embed_raw.shape[1] == metadata['embedding_dim'], 'Embedding dim mismatch'
    embed_dim_orig = X_train_embed_raw.shape[1]
    print(f'\nDataset sizes:')
    print(f'  Training:   {len(X_train_embed_raw):,} samples')
    print(f'  Validation: {len(X_val_embed_raw):,} samples')
    print(f'  Test:       {len(X_test_embed_raw):,} samples')
    print(f'  Embedding dimension: {embed_dim_orig}')
    print(f'\nFraud distribution:')
    print(f'  Training:   {y_train.sum():,.0f} fraud ({y_train.mean():.1%})')
    print(f'  Validation: {y_val.sum():,.0f} fraud ({y_val.mean():.1%})')
    print(f'  Test:       {y_test.sum():,.0f} fraud ({y_test.mean():.1%})')
    print(f'\nEmbedding statistics (train):')
    print(f'  Mean: {X_train_embed_raw.mean():.6f}')
    print(f'  Std:  {X_train_embed_raw.std():.6f}')
    print(f'  Per-dim variance (mean): {X_train_embed_raw.var(axis=0).mean():.6f}')
    print(f'  Min: {X_train_embed_raw.min():.4f}, Max: {X_train_embed_raw.max():.4f}')
    return (
        X_test_embed_raw,
        X_train_embed_raw,
        X_val_embed_raw,
        embed_dim_orig,
        n_train,
        test_row_ids,
        train_row_ids,
        val_row_ids,
        y_test,
        y_train,
        y_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2. Dimensionality Reduction (PCA)

    Reduce 512d embeddings to 64d via PCA to remove noise and reduce overfitting risk.
    """)
    return


@app.cell
def _(
    PCA,
    X_test_embed_raw,
    X_train_embed_raw,
    X_val_embed_raw,
    embed_dim_orig,
    time,
):
    PCA_DIM = 64
    print(f'PCA: {embed_dim_orig}d -> {PCA_DIM}d')
    _t0 = time.time()
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_train_embed_pca = pca.fit_transform(X_train_embed_raw)
    X_val_embed_pca = pca.transform(X_val_embed_raw)
    X_test_embed_pca = pca.transform(X_test_embed_raw)
    pca_time = time.time() - _t0
    print(f'  Explained variance: {pca.explained_variance_ratio_.sum():.2%}')
    print(f'  PCA fit+transform time: {pca_time:.1f}s')
    print(f'\nPCA complete: {PCA_DIM}d embeddings for both embeddings-only and combined models')
    return PCA_DIM, X_test_embed_pca, X_train_embed_pca, X_val_embed_pca


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Load and Align Raw Features

    Load raw tabular features and recreate the same balanced sample indices (deterministic seed) to align with the embedding splits. Raw features are then reordered using the `*_row_ids.npy` saved during embedding extraction (notebook 04), which account for the row reordering performed by `preprocess()`. Validation and test are loaded from `val_eval.parquet` / `test_eval.parquet` to match notebook 01 and embedding extraction.
    """)
    return


@app.cell
def _(
    TEMPORAL_DIR,
    cudf,
    n_train,
    np,
    test_row_ids,
    time,
    train_row_ids,
    val_row_ids,
):
    print('Loading temporal split parquets with cuDF (GPU)...')
    _t0 = time.time()
    train_gdf = cudf.read_parquet(str(TEMPORAL_DIR / 'train.parquet'))
    val_gdf = cudf.read_parquet(str(TEMPORAL_DIR / 'val_eval.parquet'))
    test_gdf = cudf.read_parquet(str(TEMPORAL_DIR / 'test_eval.parquet'))
    _load_time = time.time() - _t0
    print(f'cuDF load time: {_load_time:.2f}s')
    print('\nFeature engineering with cuDF...')
    _t0 = time.time()
    fraud_col = 'Is Fraud?'
    for gdf in [train_gdf, val_gdf, test_gdf]:
        gdf['Hour'] = gdf['Time'].str.split(':', n=1, expand=True)[0].astype(int)
        gdf['Amount'] = gdf['Amount'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)
        gdf['_target'] = ((gdf[fraud_col] == 'Yes') | (gdf[fraud_col] == '1')).astype(int)
    fe_time = time.time() - _t0
    print(f'cuDF feature engineering: {fe_time:.2f}s')
    FEATURE_COLS = ['User', 'Card', 'Year', 'Month', 'Day', 'Hour', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC']
    train_pdf = train_gdf.to_pandas()
    val_pdf = val_gdf.to_pandas()
    test_pdf = test_gdf.to_pandas()
    del train_gdf, val_gdf, test_gdf
    BALANCED_TOTAL = n_train
    fraud_mask = (train_pdf[fraud_col] == 'Yes') | (train_pdf[fraud_col] == '1')
    fraud_idx = train_pdf.index[fraud_mask].tolist()
    normal_idx = train_pdf.index[~fraud_mask].tolist()
    np.random.seed(42)
    n_fraud_target = min(len(fraud_idx), int(BALANCED_TOTAL * 0.1))
    n_normal_target = min(len(normal_idx), BALANCED_TOTAL - n_fraud_target)
    sampled_fraud = np.random.choice(fraud_idx, n_fraud_target, replace=False)
    # Recreate balanced training indices (must match embedding extraction pipeline)
    sampled_normal = np.random.choice(normal_idx, n_normal_target, replace=False)
    balanced_idx = np.concatenate([sampled_fraud, sampled_normal])
    np.random.shuffle(balanced_idx)
    print(f'\nBalanced training sample: {len(balanced_idx):,} rows ({n_fraud_target:,} fraud + {n_normal_target:,} normal)')
    X_train_raw = train_pdf.loc[balanced_idx, FEATURE_COLS].reset_index(drop=True).iloc[train_row_ids].reset_index(drop=True)
    X_val_raw = val_pdf.iloc[val_row_ids][FEATURE_COLS].reset_index(drop=True)
    X_test_raw = test_pdf.iloc[test_row_ids][FEATURE_COLS].reset_index(drop=True)
    print(f'\nRaw feature shapes (aligned with embedding splits):')
    print(f'  Train: {X_train_raw.shape}')
    print(f'  Val:   {X_val_raw.shape}')
    print(f'  Test:  {X_test_raw.shape}')
    return X_test_raw, X_train_raw, X_val_raw


@app.cell
def _(
    OrdinalEncoder,
    X_test_raw,
    X_train_raw,
    X_val_raw,
    make_column_selector,
    make_column_transformer,
    time,
):
    print('Encoding categorical features...')
    preprocessor = make_column_transformer((OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), make_column_selector(dtype_include=['object', 'category'])), remainder='passthrough')
    _t0 = time.time()
    X_train_enc = preprocessor.fit_transform(X_train_raw)
    X_val_enc = preprocessor.transform(X_val_raw)
    X_test_enc = preprocessor.transform(X_test_raw)
    enc_time = time.time() - _t0
    print(f'Encoding time: {enc_time:.2f}s')
    print(f'\nEncoded features shape:')
    print(f'  Train: {X_train_enc.shape}')
    print(f'  Val:   {X_val_enc.shape}')
    print(f'  Test:  {X_test_enc.shape}')
    return X_test_enc, X_train_enc, X_val_enc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Train XGBoost Models (HPO-Optimized)

    Three XGBoost models with independently optimized hyperparameters (Optuna, maximizing validation AP). All models use GPU-accelerated `hist` tree method and early stopping on AUC (AUROC). `scale_pos_weight` is not used -- no loss reweighting at the \~1:39 training ratio.
    """)
    return


@app.cell
def _(XGB_DEVICE, average_precision_score, roc_auc_score, time, xgb):
    XGB_PARAMS_RAW = {'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.0023, 'colsample_bytree': 0.95, 'min_child_weight': 12, 'subsample': 0.673, 'reg_alpha': 0.01, 'reg_lambda': 0.001, 'random_state': 42}
    XGB_PARAMS_EMBED = {'n_estimators': 435, 'max_depth': 12, 'learning_rate': 0.03774, 'colsample_bytree': 0.587, 'min_child_weight': 2.61, 'subsample': 0.569, 'reg_alpha': 0.01364, 'reg_lambda': 9.7e-05, 'gamma': 1.7, 'random_state': 42}
    XGB_PARAMS_COMBINED = {'n_estimators': 512, 'max_depth': 12, 'learning_rate': 0.00305, 'colsample_bytree': 0.768, 'min_child_weight': 25.85, 'subsample': 0.65, 'reg_alpha': 0.01, 'reg_lambda': 0.0001, 'gamma': 4.8, 'random_state': 42}
    scale_pos_weight = 1.0
    print(f'Class balance: scale_pos_weight = {scale_pos_weight} (no reweighting)')

    def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, params, name='Model'):
        """Train XGBoost with pre-optimized params and early stopping."""
        print(f'\nTraining {name}...')
        print(f"  Features: {X_train.shape[1]}d | Params: lr={params['learning_rate']}, depth={params['max_depth']}, mcw={params['min_child_weight']}")
        _t0 = time.time()
        clf = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, tree_method='hist', device=XGB_DEVICE, early_stopping_rounds=20, eval_metric='auc')
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        train_time = time.time() - _t0
        val_preds = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_preds)
        val_ap = average_precision_score(y_val, val_preds)
        test_preds = clf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_preds)
        test_ap = average_precision_score(y_test, test_preds)
        print(f'  Train time: {train_time:.1f}s (best_iteration={clf.best_iteration})')
        print(f'  Val  ROC-AUC: {val_auc:.4f} | AP: {val_ap:.4f}')
        print(f'  Test ROC-AUC: {test_auc:.4f} | AP: {test_ap:.4f}')
        return (clf, {'val_auc': val_auc, 'val_ap': val_ap, 'test_auc': test_auc, 'test_ap': test_ap})
    print('\n' + '=' * 60)
    print('Training XGBoost Models (HPO-optimized params)')
    print('=' * 60)
    return XGB_PARAMS_COMBINED, XGB_PARAMS_EMBED, XGB_PARAMS_RAW, train_xgb


@app.cell
def _(
    PCA_DIM,
    XGB_PARAMS_COMBINED,
    XGB_PARAMS_EMBED,
    XGB_PARAMS_RAW,
    X_test_embed_pca,
    X_test_enc,
    X_train_embed_pca,
    X_train_enc,
    X_val_embed_pca,
    X_val_enc,
    embed_dim_orig,
    np,
    train_xgb,
    y_test,
    y_train,
    y_val,
):
    n_raw = X_train_enc.shape[1]

    print(f"[1/3] BASELINE: Raw Tabular Features ({n_raw}d)")
    clf_baseline, metrics_baseline = train_xgb(
        X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test,
        params=XGB_PARAMS_RAW,
        name=f"Raw Features ({n_raw}d)"
    )

    print(f"\n[2/3] EMBEDDINGS: {PCA_DIM}d (PCA from {embed_dim_orig}d)")
    clf_embed, metrics_embed = train_xgb(
        X_train_embed_pca, y_train, X_val_embed_pca, y_val, X_test_embed_pca, y_test,
        params=XGB_PARAMS_EMBED,
        name=f"Embeddings ({PCA_DIM}d PCA)"
    )

    print(f"\n[3/3] COMBINED: Raw {n_raw}d + PCA {PCA_DIM}d Embeddings")
    X_train_combined = np.hstack([X_train_enc, X_train_embed_pca])
    X_val_combined = np.hstack([X_val_enc, X_val_embed_pca])
    X_test_combined = np.hstack([X_test_enc, X_test_embed_pca])
    clf_combined, metrics_combined = train_xgb(
        X_train_combined, y_train, X_val_combined, y_val, X_test_combined, y_test,
        params=XGB_PARAMS_COMBINED,
        name=f"Combined ({X_train_combined.shape[1]}d)"
    )
    return (
        X_train_combined,
        metrics_baseline,
        metrics_combined,
        metrics_embed,
        n_raw,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Results Summary

    Test set comparison (100K stratified samples, \~0.1% fraud). **ROC-AUC** measures overall ranking quality; **Average Precision** measures precision in the high-confidence region (operationally critical for fraud detection).
    """)
    return


@app.cell
def _(
    PCA_DIM,
    X_train_combined,
    metrics_baseline,
    metrics_combined,
    metrics_embed,
    n_raw,
):
    print('\n' + '=' * 70)
    print('RESULTS SUMMARY (Test Set - Final Holdout)')
    print('=' * 70)
    print(f"\n{'Model':<35} {'Features':>10} {'ROC-AUC':>10} {'Avg Prec':>10} {'Best At':>12}")
    print('-' * 80)
    all_aucs = [metrics_baseline['test_auc'], metrics_embed['test_auc'], metrics_combined['test_auc']]
    all_aps = [metrics_baseline['test_ap'], metrics_embed['test_ap'], metrics_combined['test_ap']]
    best_auc_idx = all_aucs.index(max(all_aucs))
    best_ap_idx = all_aps.index(max(all_aps))
    labels = [f'Raw Features (baseline)', f'{PCA_DIM}d PCA Embeddings', f'Combined ({n_raw}+{PCA_DIM}d)']
    feat_dims = [f'{n_raw}d', f'{PCA_DIM}d', f'{X_train_combined.shape[1]}d']
    metrics_list = [metrics_baseline, metrics_embed, metrics_combined]
    for _i, (label, m) in enumerate(zip(labels, metrics_list)):
        badges = []
        if _i == best_auc_idx:
            badges.append('AUC')
        if _i == best_ap_idx:
            badges.append('AP')
        badge_str = ' * ' + ','.join(badges) if badges else ''
        print(f"  {label:<33} {feat_dims[_i]:>10} {m['test_auc']:>10.4f} {m['test_ap']:>10.4f}{badge_str}")
    embed_lift_auc = (metrics_embed['test_auc'] - metrics_baseline['test_auc']) / metrics_baseline['test_auc'] * 100
    embed_lift_ap = (metrics_embed['test_ap'] - metrics_baseline['test_ap']) / metrics_baseline['test_ap'] * 100
    combined_lift_auc = (metrics_combined['test_auc'] - metrics_baseline['test_auc']) / metrics_baseline['test_auc'] * 100
    combined_lift_ap = (metrics_combined['test_ap'] - metrics_baseline['test_ap']) / metrics_baseline['test_ap'] * 100
    print('\n' + '=' * 70)
    print('LIFT OVER BASELINE (Test Set)')
    print('=' * 70)
    print(f"\n{'Model':<35} {'ROC-AUC Lift':>15} {'AP Lift':>15}")
    print('-' * 68)
    print(f"  {'Embeddings-only':<33} {('+' if embed_lift_auc > 0 else '')}{embed_lift_auc:>13.2f}% {('+' if embed_lift_ap > 0 else '')}{embed_lift_ap:>13.2f}%")
    print(f"  {'Combined':<33} {('+' if combined_lift_auc > 0 else '')}{combined_lift_auc:>13.2f}% {('+' if combined_lift_ap > 0 else '')}{combined_lift_ap:>13.2f}%")
    print('\n' + '=' * 70)
    print('KEY INSIGHT: ROC-AUC vs Average Precision')
    print('=' * 70)
    if best_auc_idx != best_ap_idx:
        print(f'\n  The two metrics disagree on the best model!')
        print(f'     Best ROC-AUC: {labels[best_auc_idx]} ({all_aucs[best_auc_idx]:.4f})')
        print(f'     Best Avg Precision: {labels[best_ap_idx]} ({all_aps[best_ap_idx]:.4f})')
        print(f'\n  For fraud detection with a fixed review budget, AP is more')
        print(f'  operationally relevant -- it directly measures how many flagged')
        print(f'  transactions are actually fraudulent.')
    else:
        print(f'\n  Both metrics agree: {labels[best_auc_idx]} is the best model.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization

    Side-by-side comparison of **ROC-AUC** (overall ranking) and **Average Precision** (fraud flagging quality).
    """)
    return


@app.cell
def _(
    OUTPUTS_DIR,
    PCA_DIM,
    metrics_baseline,
    metrics_combined,
    metrics_embed,
    n_raw,
    plt,
):
    model_labels = [f'Raw Features\n({n_raw}d)', f'Embeddings\n(PCA {PCA_DIM}d)', f'Combined\n({n_raw}+{PCA_DIM}d)']
    test_aucs = [metrics_baseline['test_auc'], metrics_embed['test_auc'], metrics_combined['test_auc']]
    test_aps = [metrics_baseline['test_ap'], metrics_embed['test_ap'], metrics_combined['test_ap']]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    best_auc_model = test_aucs.index(max(test_aucs))
    best_ap_model = test_aps.index(max(test_aps))
    bars1 = ax1.bar(model_labels, test_aucs, color=colors, edgecolor='black', linewidth=1)
    for _i, (bar, val) in enumerate(zip(bars1, test_aucs)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    bars1[best_auc_model].set_edgecolor('red')
    bars1[best_auc_model].set_linewidth(3)
    ax1.axhline(y=metrics_baseline['test_auc'], color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_ylabel('ROC-AUC Score', fontsize=12)
    ax1.set_title('ROC-AUC (Overall Ranking)', fontsize=13, fontweight='bold')
    ax1.set_ylim(min(test_aucs) * 0.95, max(test_aucs) * 1.02)
    ax1.grid(axis='y', alpha=0.3)
    bars2 = ax2.bar(model_labels, test_aps, color=colors, edgecolor='black', linewidth=1)
    for _i, (bar, val) in enumerate(zip(bars2, test_aps)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    bars2[best_ap_model].set_edgecolor('red')
    bars2[best_ap_model].set_linewidth(3)
    ax2.axhline(y=metrics_baseline['test_ap'], color='#1f77b4', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel('Average Precision', fontsize=12)
    ax2.set_title('Average Precision (Fraud Flagging Quality)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(test_aps) * 1.25)
    ax2.grid(axis='y', alpha=0.3)
    fig.suptitle('Fraud Detection: Model Comparison (Test Set)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'xgb_auc_ap_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved comparison plot to {OUTPUTS_DIR / 'xgb_auc_ap_comparison.png'}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Conclusion

    The **Combined** model (raw features + foundation model embeddings) is the clear winner. ROC-AUC is near-saturated with raw features alone, but **Average Precision is where foundation model embeddings transform the model** -- dramatically sharpening confidence in the high-precision region that matters for production fraud detection.

    **Embeddings alone underperform raw features**, as expected for decoder-only models. The 64d PCA embeddings capture sequential patterns but lose fine-grained tabular signal. The value of foundation model embeddings is as a **complement** to raw features -- they provide cross-field interaction patterns and sequential context that raw features alone cannot express.

    This blueprint demonstrated the end-to-end pipeline: dataset exploration, custom domain-specific tokenization, decoder foundation model pretraining with NeMo AutoModel, embedding extraction, and downstream XGBoost fraud detection with significant precision lift over raw features alone.
    """)
    return


if __name__ == "__main__":
    app.run()
