import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, List
from torcheeg.transforms.base_transform import EEGTransform, LabelTransform

class MeanStdNormalize(EEGTransform):
    def __init__(self, mean: Union[np.ndarray, None] = None, std: Union[np.ndarray, None] = None,
                apply_to_baseline: bool = False):
        super(MeanStdNormalize, self).__init__(apply_to_baseline=apply_to_baseline)
        self.mean = mean
        self.std = std
    def __call__(self,  *args, eeg: np.ndarray,  baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)
    def apply(self, eeg: np.ndarray, **kwargs):
        normalized_eeg = (eeg - self.mean) / (self.std + 1e-6)
        return normalized_eeg
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'mean': self.mean,'std': self.std})

class ListToTensor(LabelTransform):
    def __init__(self, dtype=torch.float):
        super(ListToTensor, self).__init__()
        self.dtype = dtype
    def __call__(self, *args, y: Union[List], **kwargs):
        return super().__call__(*args, y=y, **kwargs)
    def apply(self, y: Union[List], **kwargs):
        return torch.tensor(y, dtype=self.dtype)
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'dtype': self.dtype})

class MeanStdNormalize_grid(EEGTransform):
    def __init__(self, mean: Union[np.ndarray, None] = None, std: Union[np.ndarray, None] = None,
                 mask: Union[np.ndarray, None] = None, apply_to_baseline: bool = False):
        super(MeanStdNormalize_grid, self).__init__(apply_to_baseline=apply_to_baseline)
        self.mask = mask
        self.mean = mean
        self.std = std
    def __call__(self,  *args, eeg: np.ndarray,  baseline: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, np.ndarray]:
        return super().__call__(*args, eeg=eeg, baseline=baseline, **kwargs)
    def apply(self, eeg: np.ndarray, **kwargs):
        normalized_eeg = np.zeros_like(eeg)
        normalized_eeg[self.mask] = (eeg[self.mask] - self.mean[self.mask]) / (self.std[self.mask] + 1e-6)
        return normalized_eeg
    @property
    def repr_body(self) -> Dict:
        return dict(super().repr_body, **{'mean': self.mean, 'std': self.std, 'mask': self.mask})

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def compute_f1_score_per_label(preds, ys): # 클래스별로 TP, FP, FN을 계산
    f1_scores = {}
    labels = torch.unique(ys)
    for label in labels:
        tp = ((preds == label) & (ys == label)).sum().item()
        fp = ((preds == label) & (ys != label)).sum().item()
        fn = ((preds != label) & (ys == label)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[label.item()] = round(f1*100, 2)
    return f1_scores

def calculate_accuracy_per_label(preds, ys): # 클래스별 정확도
    accuracies = {}
    labels = torch.unique(ys)
    for label in labels:
        label_preds = preds[ys == label]
        label_true = ys[ys == label]
        correct = (label_preds == label).float().sum()
        accuracy = correct / label_true.size(0)
        accuracies[label.item()] = round(accuracy.item()*100, 2)
    return accuracies

def plot_train_result(train_losses, valid_losses, train_accs, valid_accs, path=os.getcwd(), flag=1):
    fig, loss_ax = plt.subplots(figsize=(15,6))
    acc_ax = loss_ax.twinx()

    xran = range(1, len(train_losses)+1)
    loss_ax.plot(xran, train_losses, 'y', label = 'train loss')
    loss_ax.plot(xran, valid_losses, 'r', label = 'val loss')

    acc_ax.plot(xran, train_accs, 'b', label = 'train ACC')
    acc_ax.plot(xran, valid_accs, 'g', label = 'valid ACC')

    loss_ax.set_xlabel('epoch', fontsize=15)
    loss_ax.set_ylabel('loss',fontsize=15)
    acc_ax.set_ylabel('accuracy',fontsize=15)

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(path,f'train_log_{flag}.png'), dpi=150)