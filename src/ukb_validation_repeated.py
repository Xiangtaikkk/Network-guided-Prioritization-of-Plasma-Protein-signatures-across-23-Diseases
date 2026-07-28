import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths (resolved via config.py: env vars or repo-root config.yaml) ──
from config import get_paths

_P = get_paths()
DATA_DIR   = _P["DATA_DIR"]
RESULT_DIR = _P["RESULT_DIR"]
OUT_DIR    = os.path.join(RESULT_DIR, "hpa_panel_v2")
PANEL_FILE = os.path.join(OUT_DIR, "hpa_panel_final.csv")

N_SPLITS    = 5
MIN_SAMPLES = 30
N_SEEDS     = 20
BASE_SEED   = 19          # seeds: 19, 20, ..., 38

UKB2HPA = {
    "Malignant neoplasm of cervix uteri, unspecified": "Cervical cancer",
    "Malignant neoplasm of colon, unspecified": "Colorectal cancer",
    "Malignant neoplasm of brain, unspecified": "Glioma",
    "Malignant neoplasm of brain": "Glioma",
    "Malignant neoplasm of unspecified part of bronchus or lung": "Lung cancer",
    "Malignant neoplasm of ovary": "Ovarian cancer",
    "Malignant neoplasm of pancreas, unspecified": "Pancreatic cancer",
    "Malignant neoplasm of prostate": "Prostate cancer",
    "Liver cell carcinoma": "Hepatocellular cancer",
    "Multiple myeloma": "Myeloma",
    "Benign neoplasm of pituitary gland": "Pituitary neuroendocrine tumor",
    "Unspecified type of carcinoma in situ of breast": "Breast ductal carcinoma in situ",
    "Chronic lymphocytic leukemia of B-cell type": "Chronic lymphocytic leukemia",
    "Other schizophrenia": "Schizophrenia",
    "Schizophrenia, unspecified": "Schizophrenia",
    "Paranoid schizophrenia": "Schizophrenia",
    "Catatonic schizophrenia": "Schizophrenia",
    "Residual schizophrenia": "Schizophrenia",
    "Disorganized schizophrenia": "Schizophrenia",
    "Other obesity": "Obesity",
    "Obesity, unspecified": "Obesity",
    "Other forms of systemic lupus erythematosus": "Systemic lupus erythematosus",
    "Systemic lupus erythematosus, unspecified": "Systemic lupus erythematosus",
    "Systemic lupus erythematosus with organ or system involvement": "Systemic lupus erythematosus",
    "Drug-induced systemic lupus erythematosus": "Systemic lupus erythematosus",
    "Other bipolar disorders": "Bipolar disorder",
    "Bipolar disorder, unspecified": "Bipolar disorder",
    "Rheumatoid arthritis, unspecified": "Rheumatoid arthritis",
    "Rheumatoid arthritis without rheumatoid factor": "Rheumatoid arthritis",
    "Rheumatoid arthritis without rheumatoid factor, unspecified site": "Rheumatoid arthritis",
    "Rheumatoid arthritis without rheumatoid factor, hip": "Rheumatoid arthritis",
    "Rheumatoid arthritis without rheumatoid factor, knee": "Rheumatoid arthritis",
    "Rheumatoid arthritis with rheumatoid factor, unspecified": "Rheumatoid arthritis",
    "Rheumatoid vasculitis with rheumatoid arthritis": "Rheumatoid arthritis",
    "Rheumatoid lung disease with rheumatoid arthritis": "Rheumatoid arthritis",
    "Rheumatoid lung disease with rheumatoid arthritis of unspecified site": "Rheumatoid arthritis",
    "Rheumatoid lung disease with rheumatoid arthritis of multiple sites": "Rheumatoid arthritis",
    "Other rheumatoid arthritis with rheumatoid factor of unspecified site": "Rheumatoid arthritis",
    "Other rheumatoid arthritis with rheumatoid factor of elbow": "Rheumatoid arthritis",
    "Other specified rheumatoid arthritis, elbow": "Rheumatoid arthritis",
    "Other specified rheumatoid arthritis, unspecified site": "Rheumatoid arthritis",
    "Other specified rheumatoid arthritis, multiple sites": "Rheumatoid arthritis",
    "Multiple sclerosis": "Multiple sclerosis",
    "Influenza due to certain identified influenza viruses": "Influenza",
    "Influenza due to other identified influenza virus with pneumonia": "Influenza",
    "Influenza due to other identified influenza virus with other manifestations": "Influenza",
    "Influenza due to other identified influenza virus with other respiratory manifestations": "Influenza",
    "Influenza due to unidentified influenza virus with pneumonia": "Influenza",
    "Influenza due to unidentified influenza virus with other manifestations": "Influenza",
    "Influenza due to unidentified influenza virus with other respiratory manifestations": "Influenza",
    "Pneumonia due to Streptococcus pneumoniae": "Pneumococcal pneumonia",
    "Type 2 diabetes mellitus without complications": "Type 2 diabetes",
    "Type 2 diabetes mellitus with unspecified complications": "Type 2 diabetes",
    "Type 2 diabetes mellitus with other specified complications": "Type 2 diabetes",
    "Abdominal aortic aneurysm, without rupture": "Abdominal aortic aneurysm",
    "Abdominal aortic aneurysm, ruptured": "Abdominal aortic aneurysm",
    "Thoracoabdominal aortic aneurysm, without rupture": "Abdominal aortic aneurysm",
}


def get_model(name, seed):
    if name == "LR":
        return LogisticRegression(penalty="l2", C=0.1, solver="liblinear",
                                  max_iter=200, random_state=seed)
    if name == "EN":
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  l1_ratio=0.5, C=0.1, max_iter=1000,
                                  random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=seed, n_jobs=1)
    if name == "SVM":
        return SVC(kernel="linear", C=0.1, probability=True, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=seed, verbosity=0, n_jobs=1)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=seed, n_jobs=1)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                                  random_state=seed)
    raise ValueError(name)


def build_Xy_healthy(disease, meta_dis, data_dis, data_hlt, seed):
    eids    = meta_dis[meta_dis["Disease"] == disease]["eid"].unique()
    ukb_dis = data_dis[data_dis.index.isin(eids)]
    if len(ukb_dis) < MIN_SAMPLES:   # each class must have >= MIN_SAMPLES (30)
        return None, None
    rng = np.random.RandomState(seed)
    n   = min(len(ukb_dis), len(data_hlt))
    idx = rng.choice(data_hlt.index, size=n, replace=False)
    ukb_hlt = data_hlt.loc[idx]
    common  = ukb_dis.columns.intersection(ukb_hlt.columns)
    X = pd.concat([ukb_dis[common], ukb_hlt[common]])  # NaN preserved; imputed inside CV folds
    y = np.concatenate([np.ones(len(ukb_dis)), np.zeros(len(ukb_hlt))])
    return X, y


def build_Xy_disease(disease, meta_dis, data_dis, seed):
    dis_eids   = meta_dis[meta_dis["Disease"] == disease]["eid"].unique()
    other_eids = meta_dis[meta_dis["Disease"] != disease]["eid"].unique()
    ukb_dis    = data_dis[data_dis.index.isin(dis_eids)]
    ukb_other  = data_dis[data_dis.index.isin(other_eids)]
    if len(ukb_dis) < MIN_SAMPLES:   # each class must have >= MIN_SAMPLES (30)
        return None, None
    rng = np.random.RandomState(seed)
    n   = min(len(ukb_dis), len(ukb_other))
    idx = rng.choice(ukb_other.index, size=n, replace=False)
    ukb_other_s = ukb_other.loc[idx]
    common = ukb_dis.columns.intersection(ukb_other_s.columns)
    X = pd.concat([ukb_dis[common], ukb_other_s[common]])  # NaN preserved; imputed inside CV folds
    y = np.concatenate([np.ones(len(ukb_dis)), np.zeros(len(ukb_other_s))])
    return X, y


def binary_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC":         round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else np.nan, 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else np.nan, 4),
        "F1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_one_seed(seed, panel_df, meta_dis, data_dis, data_hlt):
    rows = []
    for _, panel_row in panel_df.iterrows():
        disease    = panel_row["Disease"]
        task       = panel_row["Task"]
        model_name = panel_row["Best_Model"]
        proteins   = [p.strip() for p in str(panel_row["Proteins"]).split(";") if p.strip()]
        hpa_cv_auc = panel_row["CV_AUC"]

        if task == "vs_healthy":
            X_df, y = build_Xy_healthy(disease, meta_dis, data_dis, data_hlt, seed)
        else:
            X_df, y = build_Xy_disease(disease, meta_dis, data_dis, seed)

        if X_df is None:
            continue

        avail = [p for p in proteins if p in X_df.columns]
        if not avail:
            continue

        X_sub_raw = X_df[avail].values  # NaN preserved; imputed inside each fold
        y_arr = np.array(y)

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        fold_metrics = {k: [] for k in ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]}

        for tr_idx, val_idx in skf.split(X_sub_raw, y_arr):
            X_tr_raw, X_val_raw = X_sub_raw[tr_idx], X_sub_raw[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

            # impute: train-fold median only
            fold_med = np.nanmedian(X_tr_raw, axis=0)
            fold_med = np.where(np.isnan(fold_med), 0.0, fold_med)
            X_tr  = np.where(np.isnan(X_tr_raw),  fold_med, X_tr_raw)
            X_val = np.where(np.isnan(X_val_raw), fold_med, X_val_raw)

            sc = StandardScaler()
            X_tr_s  = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)
            m = get_model(model_name, seed)
            m.fit(X_tr_s, y_tr)
            prob = m.predict_proba(X_val_s)[:, 1]
            pred = (prob >= 0.5).astype(int)
            for k, v in binary_metrics(y_val, pred, prob).items():
                fold_metrics[k].append(v)

        rows.append({
            "Seed":        seed,
            "Disease":     disease,
            "Task":        task,
            "HPA_Model":   model_name,
            "HPA_CV_AUC":  hpa_cv_auc,
            "N_pos":       int(y_arr.sum()),
            "N_total":     len(y_arr),
            "AUC":         round(np.nanmean(fold_metrics["AUC"]),         4),
            "Accuracy":    round(np.nanmean(fold_metrics["Accuracy"]),    4),
            "Sensitivity": round(np.nanmean(fold_metrics["Sensitivity"]), 4),
            "Specificity": round(np.nanmean(fold_metrics["Specificity"]), 4),
            "F1":          round(np.nanmean(fold_metrics["F1"]),          4),
        })

    print(f"  seed={seed} done ({len(rows)} tasks)", flush=True)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    panel_df = pd.read_csv(PANEL_FILE)
    print(f"Loaded {len(panel_df)} panels from {PANEL_FILE}")

    print("Loading UKB data...")
    meta_dis = pd.read_csv(os.path.join(DATA_DIR, "ukb_meta_disease.txt"), sep="\t")
    meta_dis["Disease"] = meta_dis["Disease"].replace(UKB2HPA)
    data_dis = pd.read_csv(os.path.join(DATA_DIR, "ukb_data_disease.txt"),
                           sep="\t", index_col=0)
    data_hlt = pd.read_csv(os.path.join(DATA_DIR, "ukb_data_healthy.txt"),
                           sep="\t", index_col=0)
    print(f"  UKB disease: {len(data_dis)}  healthy: {len(data_hlt)}")

    seeds = list(range(BASE_SEED, BASE_SEED + N_SEEDS))
    print(f"Running {N_SEEDS} seeds ({seeds[0]}–{seeds[-1]}) with n_jobs=6 ...\n")

    results_nested = Parallel(n_jobs=6, backend="loky")(
        delayed(run_one_seed)(seed, panel_df, meta_dis, data_dis, data_hlt)
        for seed in seeds
    )

    raw_rows = [row for seed_rows in results_nested for row in seed_rows]
    df_raw = pd.DataFrame(raw_rows)

    grp = df_raw.groupby(["Disease", "Task", "HPA_Model", "HPA_CV_AUC", "N_pos"])
    summary_rows = []
    for (disease, task, model, hpa_auc, n_pos), g in grp:
        row = {
            "Disease":    disease,
            "Task":       task,
            "HPA_Model":  model,
            "HPA_CV_AUC": hpa_auc,
            "N_pos":      n_pos,
        }
        for metric in ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]:
            row[f"{metric}_mean"] = round(g[metric].mean(), 4)
            row[f"{metric}_std"]  = round(g[metric].std(),  4)
            row[f"{metric}_min"]  = round(g[metric].min(),  4)
            row[f"{metric}_max"]  = round(g[metric].max(),  4)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).sort_values(["Task", "Disease"])

    path_raw = os.path.join(OUT_DIR, "ukb_val_repeated_raw.csv")
    path_sum = os.path.join(OUT_DIR, "ukb_val_repeated_summary.csv")
    df_raw.to_csv(path_raw,     index=False)
    df_summary.to_csv(path_sum, index=False)
    print(f"\nSaved:")
    print(f"  {path_raw}  ({len(df_raw)} rows)")
    print(f"  {path_sum}  ({len(df_summary)} rows)")

    print(f"\n{'='*110}")
    print("Summary — UKB Repeated Validation (20 seeds)")
    print(f"{'='*110}")
    for task_label in ["vs_healthy", "vs_disease"]:
        sub = df_summary[df_summary["Task"] == task_label]
        print(f"\n  Task: {task_label}")
        print(f"  {'Disease':<43} {'Model':<8} {'N_pos':>6}  {'HPA_CV':>7}  "
              f"{'AUC_mean±std':>16}  {'AUC_min':>7}  {'AUC_max':>7}  {'Flag'}")
        print(f"  {'-'*115}")
        for _, r in sub.iterrows():
            auc_str = f"{r['AUC_mean']:.3f}±{r['AUC_std']:.3f}"
            flag = " *** AUC<0.5" if r['AUC_mean'] < 0.5 else ""
            print(f"  {r['Disease']:<43} {r['HPA_Model']:<8} {int(r['N_pos']):>6}  "
                  f"{r['HPA_CV_AUC']:>7.3f}  {auc_str:>16}  "
                  f"{r['AUC_min']:>7.3f}  {r['AUC_max']:>7.3f}{flag}")

    print(f"\n  {'='*60}")
    for task_label in ["vs_healthy", "vs_disease"]:
        sub = df_summary[df_summary["Task"] == task_label]
        print(f"  Overall {task_label}: "
              f"mean AUC = {sub['AUC_mean'].mean():.3f} ± {sub['AUC_mean'].std():.3f}  "
              f"(across {len(sub)} diseases)")

    low = df_summary[df_summary["AUC_mean"] < 0.5]
    print(f"\n  AUC_mean < 0.5: {len(low)} cases")
    for _, r in low.iterrows():
        print(f"    {r['Disease']} ({r['Task']})  AUC={r['AUC_mean']:.3f}±{r['AUC_std']:.3f}  N_pos={int(r['N_pos'])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
