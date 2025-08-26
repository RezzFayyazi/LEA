import json
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_FILES = {
    "Gemma-ideal": "./results/LEA/independence_results_entire_x_ideal_theta_ideal_y_gemma.json",
    "Gemma-generic": "./results/LEA/independence_results_entire_x_generic_theta_generic_y_gemma.json",
    "Gemma-no-retrieval": "./results/LEA/independence_results_entire_x_no_theta_y_prime_gemma.json",
}


def _extract_rag(percentages):
    if percentages is None:
        return 0.0
    if isinstance(percentages, (int, float)):
        return float(percentages)
    if not isinstance(percentages, dict):
        try:
            return float(percentages)
        except Exception:
            return 0.0

    keys_try = ["xy=True,x?y=False"] # this is the RAG key for the LEA results (as it transfers from Independent (True) to Dependent (False)
    for k in keys_try:
        if k in percentages:
            try:
                v = percentages[k]
                return float(v) if v is not None else 0.0
            except Exception:
                pass

    for k, v in percentages.items():
        k_low = k.lower()
        if any(tok in k_low for tok in ("rag", "xy", "x?y")):
            try:
                return float(v) if v is not None else 0.0
            except Exception:
                continue

    return 0.0


def load_rag_data_with_split(path, train_size=400, test_size=100, random_state=42, force_missing_to_zero=True):
    with open(path, "r") as f:
        data = json.load(f)

    items = data.items() if isinstance(data, dict) else list(enumerate(data)) if isinstance(data, list) else []
    rag = []
    missing = 0
    for _, rec in items:
        if isinstance(rec, dict):
            if "percentages" in rec:
                p = rec["percentages"]
            elif "results" in rec and isinstance(rec["results"], dict) and "percentages" in rec["results"]:
                p = rec["results"]["percentages"]
            else:
                p = rec
        else:
            p = rec

        val = _extract_rag(p)
        if isinstance(p, dict) and "xy=True,x?y=False" not in p:
            missing += 1
            if force_missing_to_zero:
                val = 0.0
        rag.append(val)

    arr = np.array(rag, dtype=float) if rag else np.array([], dtype=float)
    if arr.size and np.nanmax(arr) <= 1.0:
        arr *= 100.0
    arr = np.clip(arr, 0.0, 100.0)

    total_needed = train_size + test_size
    n = len(arr)
    if n < total_needed and total_needed > 0:
        ratio = n / total_needed
        train_size = int(train_size * ratio)
        test_size = n - train_size

    idx = np.arange(n)
    if n == 0:
        return np.array([]), np.array([])
    train_idx, test_idx = train_test_split(idx, train_size=train_size, test_size=test_size, random_state=random_state)
    return arr[train_idx], arr[test_idx]


def find_optimal_threshold_ideal_vs_others(train_dict):
    ideal_vals = []
    other_vals = []
    for name, vals in train_dict.items():
        if 'ideal' in name.lower():
            ideal_vals.append(vals)
        else:
            other_vals.append(vals)
    if not ideal_vals:
        raise ValueError("No 'ideal' model found.")
    ideal = np.concatenate(ideal_vals) if any(len(v) for v in ideal_vals) else np.array([])
    others = np.concatenate(other_vals) if any(len(v) for v in other_vals) else np.array([])

    y_true = np.concatenate([np.ones(len(ideal)), np.zeros(len(others))])
    y_scores = np.concatenate([ideal, others])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    distances = np.sqrt((fpr) ** 2 + (1 - tpr) ** 2)
    closest_idx = np.argmin(distances)
    best_threshold = thresholds[closest_idx]

    train_pred = (y_scores > best_threshold).astype(int)
    return best_threshold, {
        "roc_auc": roc_auc,
        "threshold": best_threshold,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "train_accuracy": accuracy_score(y_true, train_pred),
        "train_f1": f1_score(y_true, train_pred),
        "y_true": y_true,
        "y_scores": y_scores,
    }


def evaluate_ideal_vs_others_performance(test_dict, threshold):
    ideal_vals = []
    other_vals = []
    for name, vals in test_dict.items():
        if 'ideal' in name.lower():
            ideal_vals.append(vals)
        else:
            other_vals.append(vals)
    ideal = np.concatenate(ideal_vals) if any(len(v) for v in ideal_vals) else np.array([])
    others = np.concatenate(other_vals) if any(len(v) for v in other_vals) else np.array([])

    y_true = np.concatenate([np.ones(len(ideal)), np.zeros(len(others))])
    y_scores = np.concatenate([ideal, others])
    y_pred = (y_scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred) if y_true.size else 0.0
    prec = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) else 0.0
    rec = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores) if y_true.size else (np.array([]), np.array([]), np.array([]))
    roc_auc = auc(fpr, tpr) if fpr.size else 0.0

    cm = confusion_matrix(y_true, y_pred) if y_true.size else np.zeros((2, 2), dtype=int)
    tn, fp, fn, tp = cm.ravel()

    individual = {}
    for name, vals in test_dict.items():
        arr = np.array(vals)
        size = len(arr)
        above = int(np.sum(arr > threshold))
        individual[name] = {
            "size": size,
            "above": above,
            "below": size - above,
            "pct_above": (above / size * 100) if size else 0.0,
            "mean": float(np.mean(arr)) if size else 0.0,
        }

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
        "individual": individual,
        "scores": y_scores,
        "predictions": y_pred,
    }


def main():
    train_data, test_data = {}, {}
    for name, path in MODEL_FILES.items():
        try:
            tr, te = load_rag_data_with_split(path, train_size=400, test_size=100, random_state=42, force_missing_to_zero=True)
            train_data[name] = tr
            test_data[name] = te
            print(f"{name}: train={len(tr)} test={len(te)}")
        except Exception as e:
            print(f"{name}: error {e}")

    if len(train_data) < 2:
        print("Need at least 2 models for comparison.")
        return

    threshold, train_res = find_optimal_threshold_ideal_vs_others(train_data)
    test_res = evaluate_ideal_vs_others_performance(test_data, threshold)

    print("\nOptimal threshold (Closest to (0,1)) = {:.2f}%".format(threshold))
    print("Train ROC AUC: {:.3f}  Train Acc: {:.3f}  Train F1: {:.3f}".format(
        train_res["roc_auc"], train_res["train_accuracy"], train_res["train_f1"]))
    print("Test Accuracy: {:.3f}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}".format(
        test_res["accuracy"], test_res["precision"], test_res["recall"], test_res["f1_score"]))
    print("Sensitivity: {:.3f}  Specificity: {:.3f}  Test ROC AUC: {:.3f}".format(
        test_res["sensitivity"], test_res["specificity"], test_res["roc_auc"]))

    ideal_name = next((n for n in test_data if 'ideal' in n.lower()), None)
    if ideal_name:
        ideal_stats = test_res["individual"].get(ideal_name, {})
        print(f"\nIdeal model '{ideal_name}': {ideal_stats.get('above',0)}/{ideal_stats.get('size',0)} above threshold ({ideal_stats.get('pct_above',0):.1f}%), mean={ideal_stats.get('mean',0):.2f}%")

    others_above = sum(v["above"] for k, v in test_res["individual"].items() if 'ideal' not in k.lower())
    others_total = sum(v["size"] for k, v in test_res["individual"].items() if 'ideal' not in k.lower()) or 1
    print(f"Others above threshold: {others_above}/{others_total} ({others_above/others_total*100:.1f}%)")


if __name__ == "__main__":
    main()
