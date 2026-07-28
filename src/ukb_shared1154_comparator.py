from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from ukb_final_validation import (
    DATA_DIR,
    IMP_FILE,
    OUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    UKB2HPA,
    binary_metrics,
    build_Xy_disease,
    build_Xy_healthy,
    get_final_model,
)


FINAL_RESULTS_FILE = os.path.join(OUT_DIR, "ukb_final_results.csv")
OUT_FILE = os.path.join(OUT_DIR, "ukb_shared1154_comparator_results.csv")
EXPECTED_SHARED_N = 1154


def load_shared_proteins() -> list[str]:
    imp_df = pd.read_csv(IMP_FILE)

    counts = imp_df.groupby(["Disease", "Task"])["Protein"].nunique()
    if counts.empty:
        raise ValueError(f"No proteins found in {IMP_FILE}")
    if counts.min() != EXPECTED_SHARED_N or counts.max() != EXPECTED_SHARED_N:
        raise ValueError(
            "hpa_protein_importance.csv does not contain exactly "
            f"{EXPECTED_SHARED_N} unique proteins for every disease×task "
            f"(observed range: {counts.min()}-{counts.max()})."
        )

    first_pair = (
        imp_df.sort_values(["Disease", "Task", "Rank"])
        .groupby(["Disease", "Task"], sort=True)
        .head(EXPECTED_SHARED_N)
    )
    disease0 = first_pair.iloc[0]["Disease"]
    task0 = first_pair.iloc[0]["Task"]
    shared = (
        imp_df[(imp_df["Disease"] == disease0) & (imp_df["Task"] == task0)]
        .sort_values("Rank")["Protein"]
        .tolist()
    )

    shared_set = set(shared)
    for (disease, task), sub in imp_df.groupby(["Disease", "Task"]):
        proteins = set(sub["Protein"])
        if proteins != shared_set:
            raise ValueError(
                "Shared protein universe differs across disease×task pairs; "
                f"first mismatch: {disease} / {task}."
            )

    return shared


def load_ukb_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load UKB metadata and NPX matrices with canonical disease labels."""
    meta_dis = pd.read_csv(os.path.join(DATA_DIR, "ukb_meta_disease.txt"), sep="\t")
    meta_dis["Disease"] = meta_dis["Disease"].replace(UKB2HPA)
    data_dis = pd.read_csv(
        os.path.join(DATA_DIR, "ukb_data_disease.txt"),
        sep="\t",
        index_col=0,
    )
    data_hlt = pd.read_csv(
        os.path.join(DATA_DIR, "ukb_data_healthy.txt"),
        sep="\t",
        index_col=0,
    )
    return meta_dis, data_dis, data_hlt


def evaluate_shared1154_for_task(
    row: pd.Series,
    shared_proteins: list[str],
    meta_dis: pd.DataFrame,
    data_dis: pd.DataFrame,
    data_hlt: pd.DataFrame,
) -> dict:
    disease = row["Disease"]
    task = row["Task"]

    if task == "vs_healthy":
        X_df, y = build_Xy_healthy(disease, meta_dis, data_dis, data_hlt)
    elif task == "vs_disease":
        X_df, y = build_Xy_disease(disease, meta_dis, data_dis)
    else:
        raise ValueError(f"Unsupported task: {task}")

    if X_df is None:
        raise ValueError(f"Unable to rebuild UKB task matrix for {disease} / {task}")

    missing = [protein for protein in shared_proteins if protein not in X_df.columns]
    if missing:
        raise ValueError(
            f"{disease} / {task}: {len(missing)} of {EXPECTED_SHARED_N} shared "
            f"proteins are missing from the UKB task matrix. First missing: "
            f"{missing[:10]}"
        )

    X_raw = X_df[shared_proteins].values
    y_full = np.asarray(y)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    tr_idx, te_idx = next(sss.split(X_raw, y_full))
    X_train_raw = X_raw[tr_idx]
    X_test_raw = X_raw[te_idx]
    y_train = y_full[tr_idx]
    y_test = y_full[te_idx]

    train_med = np.nanmedian(X_train_raw, axis=0)
    train_med = np.where(np.isnan(train_med), 0.0, train_med)
    X_train = np.where(np.isnan(X_train_raw), train_med, X_train_raw)
    X_test = np.where(np.isnan(X_test_raw), train_med, X_test_raw)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_name = row["Best_Model"]
    model = get_final_model(model_name, RANDOM_STATE)
    model.fit(X_train_s, y_train)
    prob = model.predict_proba(X_test_s)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = binary_metrics(y_test, pred, prob)
    auc = round(float(roc_auc_score(y_test, prob)), 6)

    return {
        "Disease": disease,
        "Task": task,
        "N_shared_features": X_raw.shape[1],
        "N_pos": int(y_full.sum()),
        "N_test": len(y_test),
        "Best_Model": model_name,
        "Best_K": int(row["Best_K"]),
        "Panel_Test_AUC": float(row["Test_AUC"]),
        "AllProt2923_Test_AUC": float(row["AllProt_Test_AUC"]),
        "Shared1154_Test_AUC": auc,
        "Shared1154_Accuracy": metrics["Accuracy"],
        "Shared1154_Sensitivity": metrics["Sensitivity"],
        "Shared1154_Specificity": metrics["Specificity"],
        "Shared1154_F1": metrics["F1"],
    }


def main() -> None:
    final_df = pd.read_csv(FINAL_RESULTS_FILE)
    shared_proteins = load_shared_proteins()
    if len(shared_proteins) != EXPECTED_SHARED_N:
        raise ValueError(
            f"Expected {EXPECTED_SHARED_N} shared proteins, got {len(shared_proteins)}."
        )

    meta_dis, data_dis, data_hlt = load_ukb_data()

    rows = []
    for _, row in final_df.sort_values(["Task", "Disease"]).iterrows():
        result = evaluate_shared1154_for_task(
            row,
            shared_proteins,
            meta_dis,
            data_dis,
            data_hlt,
        )
        rows.append(result)
        print(
            f"{result['Disease']:45s} ({result['Task']}) "
            f"{result['Best_Model']:<8} "
            f"panel={result['Panel_Test_AUC']:.3f} "
            f"all2923={result['AllProt2923_Test_AUC']:.3f} "
            f"shared1154={result['Shared1154_Test_AUC']:.3f}",
            flush=True,
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved -> {OUT_FILE} ({len(out_df)} rows)")
    for task in ["vs_healthy", "vs_disease"]:
        sub = out_df[out_df["Task"] == task]
        print(
            f"{task}: mean Shared1154_Test_AUC = "
            f"{sub['Shared1154_Test_AUC'].mean():.6f}"
        )


if __name__ == "__main__":
    main()
