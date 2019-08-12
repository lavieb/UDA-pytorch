import context
import albumentations as albu
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..utils import get_uda_train_test_loaders
from ..utils.autoaugment import ImageNetBackwardPolicy
from ..utils.transforms import ToTensor, ToPILImage

# Transformations


def train_transform_fn(dp):
    albu_train_transform = albu.Compose([ToTensor()])
    albu_train_transform_fn = lambda x: albu_train_transform(**x)

    dp = albu_train_transform_fn(dp)

    return {'image': dp['image'],
            'mask': dp['mask']}


def test_transform_fn(dp):
    albu_test_transform = albu.Compose([ToTensor()])
    albu_test_transform_fn = lambda x: albu_test_transform(**x)

    dp = albu_test_transform_fn(dp)

    return {'image': dp['image'],
            'mask': dp['mask']}


def unsup_transform_fn(dp):
    autotransf = ImageNetBackwardPolicy()

    albu_unsup_transform = albu.Compose([ToPILImage(),
                                         autotransf,
                                         ToTensor()])
    albu_unsup_transform_fn = lambda x: albu_unsup_transform(**x)

    dp = albu_unsup_transform_fn(dp)

    return {'image': dp['image'],
            'autotransf': autotransf}


debug = True
device = "cuda"

num_epochs = 200
num_classes = 2
val_interval = 1
pred_interval = 200

# Save configuration
saves = {'save_dir': '',
         'log_dir': '',
         'model_dir': '',
         'prediction_dir': '',
         'config_dir': '',
         'save_interval': 0,
         'n_saved': 0
         }

# Datasets
train_set = None
test_set = None

# Dataloader config
batch_size = 64
num_labelled_samples = 4000
unlabelled_batch_size = 320

train_workers = 10
test_workers = 10

pin_memory = True

use_fp_16 = False

train1_sup_loader, train1_unsup_loader, train2_unsup_loader, test_loader = \
    get_uda_train_test_loaders(train_set=train_set,
                               test_set=test_set,
                               train_transform_fn=train_transform_fn,
                               test_transform_fn=test_transform_fn,
                               unsup_transform_fn=unsup_transform_fn,
                               num_classes=num_classes,
                               num_labelled_samples=num_labelled_samples,
                               batch_size=batch_size,
                               train_workers=train_workers,
                               test_workers=test_workers,
                               unlabelled_batch_size=unlabelled_batch_size,
                               pin_memory=pin_memory)

# Model
model = None

# Criterions
criterion = nn.CrossEntropyLoss()
consistency_criterion = nn.KLDivLoss(reduction='batchmean')

consistency_lambda = 1.0

# Optimizer
learning_rate = 0.03
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate,
                      momentum=0.9,
                      weight_decay=1e-4,
                      nesterov=True)

# LR scheduler
min_lr_ratio = 0.004
num_warmup_steps = 0
eta_min = learning_rate * min_lr_ratio

num_train_steps = len(train1_sup_loader) * num_epochs
lr_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=num_train_steps - num_warmup_steps)

# TSA
with_TSA = False
TSA_proba_min = 0.1
TSA_proba_max = 1.0
