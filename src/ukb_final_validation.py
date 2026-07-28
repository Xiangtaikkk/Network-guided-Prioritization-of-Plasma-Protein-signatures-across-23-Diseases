import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from config import get_paths

_P = get_paths()
DATA_DIR   = _P["DATA_DIR"]
RESULT_DIR = _P["RESULT_DIR"]
OUT_DIR    = os.path.join(RESULT_DIR, "hpa_panel_v2")
IMP_FILE   = os.path.join(OUT_DIR, "hpa_protein_importance.csv")

RANDOM_STATE = 19
TEST_SIZE    = 0.2
N_SPLITS     = 5
N_REPEATS    = 4
TOP_N        = 20
MIN_SAMPLES  = 30
MIN_VALID_TEST_AUC = 0.6
VALID_MODELS = ["LR", "EN", "RF", "SVM", "XGB", "ET", "AdaBoost"]

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

def get_cv_model(name, seed):
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
        return SVC(kernel="linear", C=0.1, probability=False, random_state=seed)
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


def get_final_model(name, seed):
    if name == "SVM":
        return SVC(kernel="linear", C=0.1, probability=True, random_state=seed)
    return get_cv_model(name, seed)


def cv_score(model, name, X_val):
    if name == "SVM":
        return model.decision_function(X_val)
    return model.predict_proba(X_val)[:, 1]


def build_Xy_healthy(disease, meta_dis, data_dis, data_hlt):
    eids    = meta_dis[meta_dis["Disease"] == disease]["eid"].unique()
    ukb_dis = data_dis[data_dis.index.isin(eids)]
    if len(ukb_dis) < MIN_SAMPLES:   # each class must have >= MIN_SAMPLES (30)
        return None, None
    rng = np.random.RandomState(RANDOM_STATE)
    n   = min(len(ukb_dis), len(data_hlt))
    idx = rng.choice(data_hlt.index, size=n, replace=False)
    ukb_hlt = data_hlt.loc[idx]
    common  = ukb_dis.columns.intersection(ukb_hlt.columns)
    X = pd.concat([ukb_dis[common], ukb_hlt[common]])  # NaN preserved; imputed after split
    y = np.concatenate([np.ones(len(ukb_dis)), np.zeros(len(ukb_hlt))])
    return X, y


def build_Xy_disease(disease, meta_dis, data_dis):
    dis_eids   = meta_dis[meta_dis["Disease"] == disease]["eid"].unique()
    other_eids = meta_dis[meta_dis["Disease"] != disease]["eid"].unique()
    ukb_dis    = data_dis[data_dis.index.isin(dis_eids)]
    ukb_other  = data_dis[data_dis.index.isin(other_eids)]
    if len(ukb_dis) < MIN_SAMPLES:   # each class must have >= MIN_SAMPLES (30)
        return None, None
    rng = np.random.RandomState(RANDOM_STATE)
    n   = min(len(ukb_dis), len(ukb_other))
    idx = rng.choice(ukb_other.index, size=n, replace=False)
    ukb_other_s = ukb_other.loc[idx]
    common = ukb_dis.columns.intersection(ukb_other_s.columns)
    X = pd.concat([ukb_dis[common], ukb_other_s[common]])  # NaN preserved; imputed after split
    y = np.concatenate([np.ones(len(ukb_dis)), np.zeros(len(ukb_other_s))])
    return X, y


def binary_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC":         round(roc_auc_score(y_true, y_prob), 6),
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(tp / (tp + fn) if (tp + fn) > 0 else np.nan, 4),
        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else np.nan, 4),
        "F1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def process_task(disease, task, proteins_top20,
                 meta_dis, data_dis, data_hlt):
    if task == "vs_healthy":
        X_df, y = build_Xy_healthy(disease, meta_dis, data_dis, data_hlt)
    else:
        X_df, y = build_Xy_disease(disease, meta_dis, data_dis)

    if X_df is None:
        print(f"  SKIP {disease} ({task}): insufficient UKB samples", flush=True)
        return None

    avail = [p for p in proteins_top20 if p in X_df.columns]
    if len(avail) < 2:
        print(f"  SKIP {disease} ({task}): <2 proteins available in UKB", flush=True)
        return None

    X_full_raw    = X_df[avail].values   # (N, n_cand), optimal panel features
    X_allprot_raw = X_df.values          # (N, ~1154), all-proteins baseline features
    y_full = np.array(y)
    n_cand = len(avail)   # ≤ 20

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                 random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sss.split(X_full_raw, y_full))
    y_train, y_test = y_full[tr_idx], y_full[te_idx]

    X_train_raw = X_full_raw[tr_idx]
    X_test_raw  = X_full_raw[te_idx]
    X_allprot_train_raw = X_allprot_raw[tr_idx]
    X_allprot_test_raw  = X_allprot_raw[te_idx]

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RANDOM_STATE)

    best_cv_auc  = -np.inf
    best_model   = None
    best_k       = None
    cv_auc_table = {}   # (model, k) → mean_auc

    for mname in VALID_MODELS:
        for k in range(1, n_cand + 1):
            fold_aucs = []
            for tr2, val in rskf.split(X_train_raw, y_train):
                X_tr2_raw = X_train_raw[tr2][:, :k]
                X_val_raw = X_train_raw[val][:, :k]
                y_tr2 = y_train[tr2]
                y_val = y_train[val]

                # impute: train-fold median only
                fold_med = np.nanmedian(X_tr2_raw, axis=0)
                fold_med = np.where(np.isnan(fold_med), 0.0, fold_med)
                X_tr2 = np.where(np.isnan(X_tr2_raw), fold_med, X_tr2_raw)
                X_val = np.where(np.isnan(X_val_raw), fold_med, X_val_raw)

                sc = StandardScaler()
                X_tr2_s = sc.fit_transform(X_tr2)
                X_val_s = sc.transform(X_val)

                m = get_cv_model(mname, RANDOM_STATE)
                m.fit(X_tr2_s, y_tr2)
                try:
                    scores = cv_score(m, mname, X_val_s)
                    auc    = roc_auc_score(y_val, scores)
                except Exception:
                    auc = np.nan
                fold_aucs.append(auc)

            mean_auc = float(np.nanmean(fold_aucs))
            cv_auc_table[(mname, k)] = round(mean_auc, 4)

            if mean_auc > best_cv_auc:
                best_cv_auc  = mean_auc
                best_model   = mname
                best_k       = k

    X_train_k_raw = X_train_raw[:, :best_k]
    X_test_k_raw  = X_test_raw[:, :best_k]
    train_med = np.nanmedian(X_train_k_raw, axis=0)
    train_med = np.where(np.isnan(train_med), 0.0, train_med)
    X_train_k = np.where(np.isnan(X_train_k_raw), train_med, X_train_k_raw)
    X_test_k  = np.where(np.isnan(X_test_k_raw),  train_med, X_test_k_raw)

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train_k)
    X_test_s  = sc.transform(X_test_k)

    m_final = get_final_model(best_model, RANDOM_STATE)
    m_final.fit(X_train_s, y_train)
    prob = m_final.predict_proba(X_test_s)[:, 1]
    pred = (prob >= 0.5).astype(int)
    test_metrics = binary_metrics(y_test, pred, prob)

    selected_proteins = avail[:best_k]

    allprot_med = np.nanmedian(X_allprot_train_raw, axis=0)
    allprot_med = np.where(np.isnan(allprot_med), 0.0, allprot_med)
    X_allprot_train = np.where(np.isnan(X_allprot_train_raw), allprot_med, X_allprot_train_raw)
    X_allprot_test  = np.where(np.isnan(X_allprot_test_raw),  allprot_med, X_allprot_test_raw)

    sc_all = StandardScaler()
    X_tr_all_s = sc_all.fit_transform(X_allprot_train)
    X_te_all_s = sc_all.transform(X_allprot_test)
    m_all = get_final_model(best_model, RANDOM_STATE)
    m_all.fit(X_tr_all_s, y_train)
    prob_all = m_all.predict_proba(X_te_all_s)[:, 1]
    allprot_auc = round(float(roc_auc_score(y_test, prob_all)), 6)

    print(f"  {disease:45s} ({task})  "
          f"{best_model:<8} K={best_k:2d}/{n_cand}  "
          f"trainCV={best_cv_auc:.3f}  "
          f"testAUC={test_metrics['AUC']:.3f}  allProt={allprot_auc:.3f}  "
          f"n={int(y_full.sum())}+{int((y_full==0).sum())}", flush=True)

    return {
        "Disease":           disease,
        "Task":              task,
        "N_allprot_features": X_allprot_raw.shape[1],
        "N_pos":             int(y_full.sum()),
        "N_test":            len(y_test),
        "HPA_Candidates":    n_cand,
        "Best_Model":        best_model,
        "Best_K":            best_k,
        "Train_CV_AUC":      round(best_cv_auc, 4),
        "AllProt_Test_AUC":  allprot_auc,
        "Test_AUC":          test_metrics["AUC"],
        "Pass_Test_AUC_Threshold": bool(test_metrics["AUC"] >= MIN_VALID_TEST_AUC),
        "Test_Accuracy":     test_metrics["Accuracy"],
        "Test_Sensitivity":  test_metrics["Sensitivity"],
        "Test_Specificity":  test_metrics["Specificity"],
        "Test_F1":           test_metrics["F1"],
        "Selected_Proteins": ";".join(selected_proteins),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    imp_df = pd.read_csv(IMP_FILE)
    top20 = (imp_df[imp_df["Rank"] <= TOP_N]
             .sort_values(["Disease", "Task", "Rank"])
             .groupby(["Disease", "Task"])["Protein"]
             .apply(list)
             .reset_index()
             .rename(columns={"Protein": "Top20"}))

    meta_dis = pd.read_csv(os.path.join(DATA_DIR, "ukb_meta_disease.txt"), sep="\t")
    meta_dis["Disease"] = meta_dis["Disease"].replace(UKB2HPA)
    data_dis = pd.read_csv(os.path.join(DATA_DIR, "ukb_data_disease.txt"),
                           sep="\t", index_col=0)
    data_hlt = pd.read_csv(os.path.join(DATA_DIR, "ukb_data_healthy.txt"),
                           sep="\t", index_col=0)

    results = Parallel(n_jobs=6, backend="loky")(
        delayed(process_task)(
            row["Disease"], row["Task"], row["Top20"],
            meta_dis, data_dis, data_hlt
        )
        for _, row in top20.iterrows()
    )

    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows).sort_values(["Task", "Disease"]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "ukb_final_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}  ({len(df)} rows)")

    print(f"\n{'='*115}")
    print("Summary — UKB Final Validation (HPA top-20 candidates, UKB-train selects model+K, UKB-test evaluates)")
    print(f"{'='*115}")

    for task_label in ["vs_healthy", "vs_disease"]:
        sub = df[df["Task"] == task_label]
        print(f"\n  Task: {task_label}  (N={len(sub)})")
        print(f"  {'Disease':<43} {'Model':<8} {'K':>3}  {'N_pos':>6}  "
              f"{'trainCV':>8}  {'testAUC':>8}  {'Acc':>6}  {'Sen':>6}  {'Spe':>6}  {'F1':>6}  Flag")
        print(f"  {'-'*120}")
        for _, r in sub.iterrows():
            flag = f" *** AUC<{MIN_VALID_TEST_AUC:.1f}" if r["Test_AUC"] < MIN_VALID_TEST_AUC else ""
            print(f"  {r['Disease']:<43} {r['Best_Model']:<8} {int(r['Best_K']):>3}  "
                  f"{int(r['N_pos']):>6}  "
                  f"{r['Train_CV_AUC']:>8.3f}  {r['Test_AUC']:>8.3f}  "
                  f"{r['Test_Accuracy']:>6.3f}  {r['Test_Sensitivity']:>6.3f}  "
                  f"{r['Test_Specificity']:>6.3f}  {r['Test_F1']:>6.3f}{flag}")

    print(f"\n  {'='*60}")
    for task_label in ["vs_healthy", "vs_disease"]:
        sub = df[df["Task"] == task_label]
        print(f"  Overall {task_label}: mean testAUC = {sub['Test_AUC'].mean():.3f} ± {sub['Test_AUC'].std():.3f}")

    low = df[df["Test_AUC"] < MIN_VALID_TEST_AUC]
    print(f"\n  Downstream-excluded panels (Test_AUC < {MIN_VALID_TEST_AUC:.1f}): {len(low)} cases")
    for _, r in low.iterrows():
        print(f"    {r['Disease']} ({r['Task']})  AUC={r['Test_AUC']:.3f}  N_pos={int(r['N_pos'])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
