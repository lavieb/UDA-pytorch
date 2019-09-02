import numpy as np

from torch.utils.data import Subset, Dataset, DataLoader
import torch

from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

from utils import autoaugment
from utils.transforms import RandomErasing, ToTensor


def set_seed(seed):

    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_uda_train_test_loaders(train_set,
                               test_set,
                               train_transform_fn,
                               test_transform_fn,
                               unsup_transform_bf_fn,
                               unsup_transform_af_fn,
                               unsup_transform_pl_fn,
                               num_classes,
                               num_labelled_samples,
                               batch_size,
                               train_workers,
                               test_workers,
                               unsup_set=None,
                               unlabelled_batch_size=None,
                               pin_memory=True,
                               shuffle=False):

    if unsup_set is None:
        train1_set, train2_set = stratified_train_labelled_unlabelled_split(train_set,
                                                                            num_labelled_samples=num_labelled_samples,
                                                                            num_classes=num_classes, seed=12)
    else:
        train1_set = Subset(train_set, np.random.permutation(len(train_set))[:num_labelled_samples])
        train2_set = unsup_set

    train1_sup_ds = TransformedDataset(train1_set, train_transform_fn)
    test_ds = TransformedDataset(test_set, test_transform_fn)

    original_transform = unsup_transform_bf_fn
    augmentation_transform = unsup_transform_af_fn
    train1_unsup_ds = TransformedDataset(train1_set, UDATransform(original_transform, augmentation_transform, unsup_transform_pl_fn))
    train2_unsup_ds = TransformedDataset(train2_set, UDATransform(original_transform, augmentation_transform, unsup_transform_pl_fn))

    if unlabelled_batch_size is None:
        unlabelled_batch_size = batch_size

    train1_sup_loader = DataLoader(train1_sup_ds, batch_size=batch_size, shuffle=shuffle, num_workers=train_workers, pin_memory=pin_memory)
    train1_unsup_loader = DataLoader(train1_unsup_ds, batch_size=unlabelled_batch_size, shuffle=shuffle, num_workers=train_workers, pin_memory=pin_memory)
    train2_unsup_loader = DataLoader(train2_unsup_ds, batch_size=unlabelled_batch_size, shuffle=shuffle, num_workers=train_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=True, num_workers=test_workers, pin_memory=pin_memory)

    return train1_sup_loader, train1_unsup_loader, train2_unsup_loader, test_loader


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        dp = self.transform_fn(dp)
        return dp[0], dp[1]


class UDATransform:

    def __init__(self, before_transform, policy_transform, after_transform, copy=False):
        self.before_transform = before_transform
        self.policy_transform = policy_transform
        self.after_transform = after_transform
        self.copy = copy

    def __call__(self, dp):        
        if self.copy:
            aug_dp = dp.copy()
        else:
            aug_dp = dp
        tdp1 = self.before_transform(aug_dp)
        tdp2 = self.policy_transform(tdp1)

        tdp1 = self.after_transform(tdp1)
        tdp2 = self.after_transform(tdp2)

        return tdp1[0], tdp2[0]


def stratified_train_labelled_unlabelled_split(ds, num_labelled_samples, num_classes, seed=None):
    labelled_indices = []
    unlabelled_indices = []
    
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(len(ds))
    
    class_counters = list([0] * num_classes)
    max_counter = num_labelled_samples // num_classes
    for i in indices:
        dp = ds[i]        
        
        if num_labelled_samples < sum(class_counters):
            unlabelled_indices.append(i)
        else:
            y = dp[1]        
            c = class_counters[y]
            if c < max_counter:
                class_counters[y] += 1
                labelled_indices.append(i)
            else:
                unlabelled_indices.append(i)

    assert len(set(labelled_indices) & set(unlabelled_indices)) == 0, \
        "{}".format(set(labelled_indices) & set(unlabelled_indices))
    
    train_labelled_ds = Subset(ds, labelled_indices)
    train_unlabelled_ds = Subset(ds, unlabelled_indices)
    return train_labelled_ds, train_unlabelled_ds
