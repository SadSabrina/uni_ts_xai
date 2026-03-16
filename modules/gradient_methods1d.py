import torch
import numpy as np
from tqdm import tqdm

def vanilla_gradients(model, input_tensor, target_class):
    # Создаем копию для градиентов
    input_copy = input_tensor.clone().detach()
    input_copy.requires_grad = True # заставляем вовзращать градиетны

    output = model(input_copy)
    target = output[0, target_class]
    model.zero_grad()
    target.backward() # делаем бэкпроп

    gradients = input_copy.grad.data.cpu().numpy()[0]
    saliency_map = np.max(np.abs(gradients), axis=0)

    return saliency_map

def smoothgrad(model, input_tensor, target_class, n_samples=50, noise_level=0.15):

    smooth_grad = torch.zeros_like(input_tensor) # заглушка для карты, в которую будем складывать шум

    for _ in tqdm(range(n_samples), desc="Smoothgrad progress"):

        noise = torch.randn_like(input_tensor) * noise_level # шум для входной картинки
        noisy_input = (input_tensor + noise).clone().detach()
        noisy_input.requires_grad = True

        # шаги, аналогчиные стандартной SalMap
        output = model(noisy_input)
        target = output[0, target_class]

        model.zero_grad()
        target.backward()

        smooth_grad += noisy_input.grad

    smooth_grad = smooth_grad / n_samples
    smooth_grad = smooth_grad.detach().cpu().numpy()[0]
    saliency_map = np.max(np.abs(smooth_grad), axis=0)

    return saliency_map