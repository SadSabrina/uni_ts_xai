from tqdm.notebook import tqdm
import numpy as np
import torch
from uni_ts_xai.modules.metrics import curve_auc, insertion_metric_1d, deletion_metric_1d, prepare_saliency_1d
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR


def get_saliency_map_tsinterpret(model, item_1ft, label, method, use_tsr, num_timesteps, num_features):
    """
    item_1ft: numpy array shape (1, F, T)
    """
    int_mod = TSR(
        model,
        num_timesteps,
        num_features,
        method=method,
        mode="feat",
        tsr=use_tsr,
    )

    exp = int_mod.explain(
        item_1ft,
        labels=int(label),
        TSR=use_tsr,
        assignment=0.0
    )

    exp = np.array(exp).squeeze()

    if exp.ndim != 1:
        raise ValueError(f"Ожидалась 1D карта, получено {exp.shape}")

    return np.abs(exp.astype(float))


def topk_overlap_metrics(saliency_map, reference_idx, top_k=150):
    saliency_map = np.array(saliency_map).squeeze()
    pred_top = np.argsort(saliency_map)[::-1][:top_k]

    pred_set = set(pred_top.tolist())
    ref_set = set(reference_idx.tolist()) if not isinstance(reference_idx, set) else reference_idx

    intersection = len(pred_set & ref_set)
    union = len(pred_set | ref_set)

    recall = intersection / len(ref_set)
    precision = intersection / len(pred_set)
    jaccard = intersection / union if union > 0 else 0.0

    return {
        "intersection": intersection,
        "recall": recall,
        "precision": precision,
        "jaccard": jaccard,
        "top_idx": pred_top,
    }

   
def evaluate_method_on_dataset(
    model,
    items_matrix,              # shape (N, T)
    pred_labels,
    true_labels,
    method,
    use_tsr,
    num_timesteps,
    num_features,
    top_mean_idx,
    top_median_idx,
    steps=50,
    baseline_value=0.0,
    mode="feat",
    window_size=1,
    target_mode="pred",        # "pred" или "true"
):
    deletion_curves = []
    insertion_curves = []

    overlaps_mean = []
    overlaps_median = []

    saliency_maps = []

    for i in tqdm(range(len(items_matrix)), desc='Method in progress'):
        signal = items_matrix[i]                  # (T,)
        item = signal[None, None, :]              # (1, 1, T)

        if target_mode == "pred":
            label = int(pred_labels[i])
        elif target_mode == "true":
            label = int(true_labels[i])
        else:
            raise ValueError("target_mode должен быть 'pred' или 'true'")

        smap = get_saliency_map_tsinterpret(
            model=model,
            item_1ft=item,
            label=label,
            method=method,
            use_tsr=use_tsr,
            num_timesteps=num_timesteps,
            num_features=num_features,
        )
        saliency_maps.append(smap)

        # overlap с mean/median reference
        ov_mean = topk_overlap_metrics(smap, top_mean_idx, top_k=150)
        ov_median = topk_overlap_metrics(smap, top_median_idx, top_k=150)

        overlaps_mean.append(ov_mean)
        overlaps_median.append(ov_median)

        # insertion / deletion
        input_tensor = torch.tensor(item, dtype=torch.float32, device=next(model.parameters()).device)

        del_curve = deletion_metric_1d(
            model=model,
            input_tensor=input_tensor,
            saliency_map=smap,
            target_class=label,
            steps=steps,
            baseline_value=baseline_value,
            mode=mode,
            window_size=window_size,
        )
        ins_curve = insertion_metric_1d(
            model=model,
            input_tensor=input_tensor,
            saliency_map=smap,
            target_class=label,
            steps=steps,
            baseline_value=baseline_value,
            mode=mode,
            window_size=window_size,
        )

        deletion_curves.append(del_curve)
        insertion_curves.append(ins_curve)

    deletion_curves = np.stack(deletion_curves, axis=0)   # (N, steps+1)
    insertion_curves = np.stack(insertion_curves, axis=0)
    saliency_maps = np.stack(saliency_maps, axis=0)       # (N, T)

    result = {
        "method_name": f"{method}+TSR" if use_tsr else method,
        "saliency_maps": saliency_maps,

        "deletion_curves_all": deletion_curves,
        "insertion_curves_all": insertion_curves,
        "deletion_curve_mean": deletion_curves.mean(axis=0),
        "insertion_curve_mean": insertion_curves.mean(axis=0),
        "deletion_auc_mean": np.mean([curve_auc(c) for c in deletion_curves]),
        "insertion_auc_mean": np.mean([curve_auc(c) for c in insertion_curves]),

        "mean_overlap_intersection_avg": np.mean([x["intersection"] for x in overlaps_mean]),
        "mean_overlap_recall_avg": np.mean([x["recall"] for x in overlaps_mean]),
        "mean_overlap_precision_avg": np.mean([x["precision"] for x in overlaps_mean]),
        "mean_overlap_jaccard_avg": np.mean([x["jaccard"] for x in overlaps_mean]),

        "median_overlap_intersection_avg": np.mean([x["intersection"] for x in overlaps_median]),
        "median_overlap_recall_avg": np.mean([x["recall"] for x in overlaps_median]),
        "median_overlap_precision_avg": np.mean([x["precision"] for x in overlaps_median]),
        "median_overlap_jaccard_avg": np.mean([x["jaccard"] for x in overlaps_median]),
    }

    return result

def evaluate_fixed_map_on_dataset(
    model,
    items_matrix,              # shape (N, T)
    pred_labels,
    true_labels,
    fixed_map,                 # shape (T,)
    method_name,
    top_mean_idx,
    top_median_idx,
    steps=50,
    baseline_value=0.0,
    mode="feat",
    window_size=5,
    target_mode="pred",        # "pred" или "true"
):
    deletion_curves = []
    insertion_curves = []

    overlaps_mean = []
    overlaps_median = []

    fixed_map = prepare_saliency_1d(fixed_map)

    for i in range(len(items_matrix)):
        signal = items_matrix[i]                  # (T,)
        item = signal[None, None, :]              # (1, 1, T)

        if target_mode == "pred":
            label = int(pred_labels[i])
        elif target_mode == "true":
            label = int(true_labels[i])
        else:
            raise ValueError("target_mode должен быть 'pred' или 'true'")

        input_tensor = torch.tensor(
            item,
            dtype=torch.float32,
            device=next(model.parameters()).device
        )

        # overlap
        ov_mean = topk_overlap_metrics(fixed_map, top_mean_idx, top_k=150)
        ov_median = topk_overlap_metrics(fixed_map, top_median_idx, top_k=150)

        overlaps_mean.append(ov_mean)
        overlaps_median.append(ov_median)

        # deletion / insertion
        del_curve = deletion_metric_1d(
            model=model,
            input_tensor=input_tensor,
            saliency_map=fixed_map,
            target_class=label,
            steps=steps,
            baseline_value=baseline_value,
            mode=mode,
            window_size=window_size,
        )

        ins_curve = insertion_metric_1d(
            model=model,
            input_tensor=input_tensor,
            saliency_map=fixed_map,
            target_class=label,
            steps=steps,
            baseline_value=baseline_value,
            mode=mode,
            window_size=window_size,
        )

        deletion_curves.append(del_curve)
        insertion_curves.append(ins_curve)

    deletion_curves = np.stack(deletion_curves, axis=0)
    insertion_curves = np.stack(insertion_curves, axis=0)

    result = {
        "method_name": method_name,
        "saliency_maps": np.repeat(fixed_map[None, :], len(items_matrix), axis=0),

        "deletion_curves_all": deletion_curves,
        "insertion_curves_all": insertion_curves,
        "deletion_curve_mean": deletion_curves.mean(axis=0),
        "insertion_curve_mean": insertion_curves.mean(axis=0),
        "deletion_auc_mean": np.mean([curve_auc(c) for c in deletion_curves]),
        "insertion_auc_mean": np.mean([curve_auc(c) for c in insertion_curves]),

        "mean_overlap_intersection_avg": np.mean([x["intersection"] for x in overlaps_mean]),
        "mean_overlap_recall_avg": np.mean([x["recall"] for x in overlaps_mean]),
        "mean_overlap_precision_avg": np.mean([x["precision"] for x in overlaps_mean]),
        "mean_overlap_jaccard_avg": np.mean([x["jaccard"] for x in overlaps_mean]),

        "median_overlap_intersection_avg": np.mean([x["intersection"] for x in overlaps_median]),
        "median_overlap_recall_avg": np.mean([x["recall"] for x in overlaps_median]),
        "median_overlap_precision_avg": np.mean([x["precision"] for x in overlaps_median]),
        "median_overlap_jaccard_avg": np.mean([x["jaccard"] for x in overlaps_median]),
    }

    return result