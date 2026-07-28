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
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from config import get_paths

_P = get_paths()
HPA_DATA   = _P["HPA_DATA"]
HPA_META   = _P["HPA_META"]
UKB_DATA   = _P["UKB_DATA"]
RESULT_DIR = _P["RESULT_DIR"]
OUT_DIR    = os.path.join(RESULT_DIR, "hpa_panel_v2")

RANDOM_STATE = 19
N_SPLITS     = 5
TOP_N        = 20
MIN_SAMPLES  = 10   # HDBA per-disease sample sizes are small
VALID_MODELS = ["LR", "EN", "RF", "SVM", "XGB", "ET", "AdaBoost"]

# The 23 analysed diseases (public; also listed in the manuscript Table S1).
DISEASES = [
    "Abdominal aortic aneurysm",
    "Bipolar disorder",
    "Breast ductal carcinoma in situ",
    "Cervical cancer",
    "Chronic lymphocytic leukemia",
    "Colorectal cancer",
    "Diffuse large B-cell lymphoma",
    "Glioma",
    "Hepatocellular cancer",
    "Influenza",
    "Lung cancer",
    "Multiple sclerosis",
    "Myeloma",
    "Obesity",
    "Ovarian cancer",
    "Pancreatic cancer",
    "Pituitary neuroendocrine tumor",
    "Pneumococcal pneumonia",
    "Prostate cancer",
    "Rheumatoid arthritis",
    "Schizophrenia",
    "Systemic lupus erythematosus",
    "Type 2 diabetes",
]


def load_disease_list():
    return sorted(set(DISEASES))

def get_model(name, for_cv=False):

    if name == "LR":
        return LogisticRegression(penalty="l2", C=0.1, solver="liblinear",
                                  max_iter=200, random_state=RANDOM_STATE)
    if name == "EN":
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  l1_ratio=0.5, C=0.1, max_iter=1000,
                                  random_state=RANDOM_STATE)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=1)
    if name == "SVM":
        return SVC(kernel="linear", C=0.1,
                   probability=(not for_cv),
                   random_state=RANDOM_STATE)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=1)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                                  random_state=RANDOM_STATE)
    raise ValueError(name)


def get_cv_scores(model, name, X_val):
    if name == "SVM":
        return model.decision_function(X_val)
    return model.predict_proba(X_val)[:, 1]


def extract_importance(model, name, X_train_sc, y_train):
    if name in ("LR", "EN", "SVM"):
        return np.abs(model.coef_[0])
    if name in ("RF", "XGB", "AdaBoost", "ET"):
        return model.feature_importances_
    raise ValueError(f"Unknown model for importance extraction: {name}")


def run_cv(X, y):
    n_feat  = X.shape[1]
    n_avail = min(TOP_N, n_feat)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {}

    for name in VALID_MODELS:
        fold_imps   = []                              # shape: (n_folds, n_feat)
        fold_k_aucs = [[] for _ in range(n_avail)]   # [k] → list of AUC per fold

        for tr_idx, val_idx in skf.split(X, y):
            X_tr_raw, X_val_raw = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # impute: train-fold median only → apply to val fold (no leakage)
            fold_med = np.nanmedian(X_tr_raw, axis=0)
            fold_med = np.where(np.isnan(fold_med), 0.0, fold_med)
            X_tr  = np.where(np.isnan(X_tr_raw),  fold_med, X_tr_raw)
            X_val = np.where(np.isnan(X_val_raw), fold_med, X_val_raw)

            sc      = StandardScaler()
            X_tr_s  = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)
            m_full = get_model(name, for_cv=True)
            m_full.fit(X_tr_s, y_tr)
            imp  = extract_importance(m_full, name, X_tr_s, y_tr)
            fold_imps.append(imp)
            fold_rank = np.argsort(imp)[::-1]   

            for k in range(1, n_avail + 1):
                top_idx  = fold_rank[:k]
                X_tr_k   = X_tr_s[:, top_idx]
                X_val_k  = X_val_s[:, top_idx]
                m_k = get_model(name, for_cv=True)
                m_k.fit(X_tr_k, y_tr)
                try:
                    scores = get_cv_scores(m_k, name, X_val_k)
                    auc    = roc_auc_score(y_val, scores)
                except Exception:
                    auc = np.nan
                fold_k_aucs[k - 1].append(auc)

        mean_imp    = np.mean(fold_imps, axis=0)
        mean_k_aucs = [np.nanmean(v) for v in fold_k_aucs]
        std_k_aucs  = [np.nanstd(v)  for v in fold_k_aucs]

        optimal_k = int(np.nanargmax(mean_k_aucs)) + 1
        final_rank = np.argsort(mean_imp)[::-1]  

        results[name] = {
            "optimal_k":          optimal_k,
            "cv_auc":             round(mean_k_aucs[optimal_k - 1], 4),
            "cv_auc_std":         round(std_k_aucs[optimal_k - 1],  4),
            "mean_k_aucs":        [round(v, 4) for v in mean_k_aucs],
            "std_k_aucs":         [round(v, 4) for v in std_k_aucs],
            "mean_importance":    mean_imp,
            "final_ranked_indices": final_rank,
        }

    return results

def build_Xy_healthy(disease, hpa_meta_dis, hpa_data_dis, hpa_data_hlt):
    eids = hpa_meta_dis[hpa_meta_dis["Disease"] == disease]["DAid"].tolist()
    dis  = hpa_data_dis[hpa_data_dis.index.isin(eids)]
    hlt  = hpa_data_hlt
    if len(dis) < MIN_SAMPLES:
        return None, None, None
    common = dis.columns.intersection(hlt.columns).tolist()
    X = pd.concat([dis[common], hlt[common]], axis=0)  # NaN preserved; imputed inside CV folds
    y = np.concatenate([np.ones(len(dis)), np.zeros(len(hlt))])
    return X.values, y, common


def build_Xy_disease(disease, hpa_meta_dis, hpa_data_dis):
    dis_eids   = hpa_meta_dis[hpa_meta_dis["Disease"] == disease]["DAid"].tolist()
    other_eids = hpa_meta_dis[hpa_meta_dis["Disease"] != disease]["DAid"].tolist()
    dis   = hpa_data_dis[hpa_data_dis.index.isin(dis_eids)]
    other = hpa_data_dis[hpa_data_dis.index.isin(other_eids)]
    if len(dis) < MIN_SAMPLES or len(other) < MIN_SAMPLES:
        return None, None, None
    common = dis.columns.intersection(other.columns).tolist()
    X = pd.concat([dis[common], other[common]], axis=0)  # NaN preserved; imputed inside CV folds
    y = np.concatenate([np.ones(len(dis)), np.zeros(len(other))])
    return X.values, y, common

def process_one_task(disease, task, hpa_meta_dis, hpa_data_dis, hpa_data_hlt):
    if task == "healthy":
        X, y, feat = build_Xy_healthy(disease, hpa_meta_dis, hpa_data_dis, hpa_data_hlt)
    else:
        X, y, feat = build_Xy_disease(disease, hpa_meta_dis, hpa_data_dis)

    task_label = "vs_healthy" if task == "healthy" else "vs_disease"

    if X is None:
        print(f"  SKIP {disease} ({task_label}): insufficient samples", flush=True)
        return None

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    cv_res = run_cv(X, y)

    best_model = max(cv_res, key=lambda m: cv_res[m]["cv_auc"])
    best = cv_res[best_model]

    final_proteins = [feat[i] for i in best["final_ranked_indices"][:best["optimal_k"]]]

    summary_row = {
        "Disease":    disease,
        "Task":       task_label,
        "N_pos":      n_pos,
        "N_neg":      n_neg,
        "Best_Model": best_model,
        "Optimal_K":  best["optimal_k"],
        "CV_AUC":     best["cv_auc"],
        "CV_AUC_std": best["cv_auc_std"],
        "Proteins":   ";".join(final_proteins),
    }

    all_model_rows = []
    for mname, mres in cv_res.items():
        prots = [feat[i] for i in mres["final_ranked_indices"][:mres["optimal_k"]]]
        all_model_rows.append({
            "Disease":    disease,
            "Task":       task_label,
            "Model":      mname,
            "Optimal_K":  mres["optimal_k"],
            "CV_AUC":     mres["cv_auc"],
            "CV_AUC_std": mres["cv_auc_std"],
            "Proteins":   ";".join(prots),
        })

    imp_rows = []
    for rank, idx in enumerate(best["final_ranked_indices"], 1):
        imp_rows.append({
            "Disease":    disease,
            "Task":       task_label,
            "Model":      best_model,
            "Rank":       rank,
            "Protein":    feat[idx],
            "Mean_Importance": round(float(best["mean_importance"][idx]), 6),
            "In_Panel":   rank <= best["optimal_k"],
        })

    print(f"  {disease:45s} ({task_label})  "
          f"{best_model:<8} K={best['optimal_k']:2d}  "
          f"CV_AUC={best['cv_auc']:.3f}±{best['cv_auc_std']:.3f}  "
          f"n={n_pos}+{n_neg}", flush=True)

    return summary_row, all_model_rows, imp_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading HPA data...")
    hpa_meta_full = pd.read_csv(HPA_META, sep="\t")
    hpa_data_full = pd.read_csv(HPA_DATA, sep="\t", index_col=0)

    ukb_proteins  = pd.read_csv(UKB_DATA, sep="\t", index_col=0, nrows=1).columns.tolist()
    common_prots  = [c for c in hpa_data_full.columns if c in set(ukb_proteins)]
    hpa_data_full = hpa_data_full[common_prots]
    hpa_meta_dis = hpa_meta_full[hpa_meta_full["Disease"] != "Healthy"].copy()
    hpa_meta_hlt = hpa_meta_full[hpa_meta_full["Disease"] == "Healthy"].copy()
    hpa_data_dis = hpa_data_full.loc[hpa_data_full.index.isin(hpa_meta_dis["DAid"])]
    hpa_data_hlt = hpa_data_full.loc[hpa_data_full.index.isin(hpa_meta_hlt["DAid"])]

    diseases = load_disease_list()
    tasks = [(d, t) for d in diseases for t in ["healthy", "disease"]]

    raw_results = Parallel(n_jobs=6, backend="loky")(
        delayed(process_one_task)(
            disease, task, hpa_meta_dis, hpa_data_dis, hpa_data_hlt
        )
        for disease, task in tasks
    )

    summary_rows   = []
    all_model_rows = []
    imp_rows       = []
    for res in raw_results:
        if res is None:
            continue
        s_row, am_rows, i_rows = res
        summary_rows.append(s_row)
        all_model_rows.extend(am_rows)
        imp_rows.extend(i_rows)

    df_final = pd.DataFrame(summary_rows)
    df_all   = pd.DataFrame(all_model_rows)
    df_imp   = pd.DataFrame(imp_rows)

    path_final = os.path.join(OUT_DIR, "hpa_panel_final.csv")
    path_all   = os.path.join(OUT_DIR, "hpa_all_models_cv.csv")
    path_imp   = os.path.join(OUT_DIR, "hpa_protein_importance.csv")

    df_final.to_csv(path_final, index=False)
    df_all.to_csv(path_all,     index=False)
    df_imp.to_csv(path_imp,     index=False)

    print(f"\nSaved:")
    print(f"  {path_final}")
    print(f"  {path_all}")
    print(f"  {path_imp}")
    print("\nDone.")


if __name__ == "__main__":
    main()
