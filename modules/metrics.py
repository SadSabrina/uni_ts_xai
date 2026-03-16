import numpy as np
import torch 
import torch.nn.functional as F

def prepare_saliency_1d(smap):
    smap = np.array(smap).squeeze()
    if smap.ndim != 1:
        raise ValueError(f"Ожидалась 1D карта после squeeze(), получено shape={smap.shape}")
    return np.abs(smap.astype(float))


def deletion_metric_1d(
    model,
    input_tensor,
    saliency_map,
    target_class,
    steps=50,
    baseline_value=0.0,
    mode="feat",
    window_size=1,
):
    model.eval()

    saliency_map = prepare_saliency_1d(saliency_map)
    T = len(saliency_map)

    order = np.argsort(saliency_map)[::-1].copy()
    scores = []

    for step in range(steps + 1):
        n_delete = int(T * step / steps)
        chosen = order[:n_delete]

        mask = np.zeros(T, dtype=bool)
        for idx in chosen:
            left = max(0, idx - window_size // 2)
            right = min(T, idx + window_size // 2 + 1)
            mask[left:right] = True

        idx_to_delete = torch.as_tensor(
            np.where(mask)[0],
            dtype=torch.long,
            device=input_tensor.device
        )

        temp_input = input_tensor.clone()

        if temp_input.ndim == 3:
            if mode == "feat":
                temp_input[0, :, idx_to_delete] = baseline_value
            elif mode == "time":
                temp_input[0, idx_to_delete, :] = baseline_value
            else:
                raise ValueError("mode должен быть 'feat' или 'time'")
        elif temp_input.ndim == 2:
            temp_input[0, idx_to_delete] = baseline_value
        else:
            raise ValueError(f"Неподдерживаемая форма input_tensor: {temp_input.shape}")

        with torch.no_grad():
            output = model(temp_input)
            prob = F.softmax(output)[0, target_class].item()
            scores.append(prob)

    return np.array(scores)


def insertion_metric_1d(
    model,
    input_tensor,
    saliency_map,
    target_class,
    steps=50,
    baseline_value=0.0,
    mode="feat",
    window_size=1,
):
    model.eval()

    saliency_map = prepare_saliency_1d(saliency_map)
    T = len(saliency_map)

    order = np.argsort(saliency_map)[::-1].copy()
    scores = []

    baseline = torch.ones_like(input_tensor) * baseline_value

    for step in range(steps + 1):
        n_insert = int(T * step / steps)
        chosen = order[:n_insert]

        mask = np.zeros(T, dtype=bool)
        for idx in chosen:
            left = max(0, idx - window_size // 2)
            right = min(T, idx + window_size // 2 + 1)
            mask[left:right] = True

        idx_to_insert = torch.as_tensor(
            np.where(mask)[0],
            dtype=torch.long,
            device=input_tensor.device
        )

        temp_input = baseline.clone()

        if temp_input.ndim == 3:
            if mode == "feat":
                temp_input[0, :, idx_to_insert] = input_tensor[0, :, idx_to_insert]
            elif mode == "time":
                temp_input[0, idx_to_insert, :] = input_tensor[0, idx_to_insert, :]
            else:
                raise ValueError("mode должен быть 'feat' или 'time'")
        elif temp_input.ndim == 2:
            temp_input[0, idx_to_insert] = input_tensor[0, idx_to_insert]
        else:
            raise ValueError(f"Неподдерживаемая форма input_tensor: {input_tensor.shape}")

        with torch.no_grad():
            output = model(temp_input)
            prob = F.softmax(output)[0, target_class].item()
            scores.append(prob)

    return np.array(scores)


def curve_auc(curve):
    x = np.linspace(0, 1, len(curve))
    return np.trapz(curve, x)

