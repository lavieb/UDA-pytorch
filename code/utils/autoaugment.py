# Code taken from https://github.com/DeepVoltaire/AutoAugment
# Code taken from https://github.com/4uiiurz1/pytorch-auto-augment

from abc import abstractstaticmethod
from PIL import Image, ImageEnhance, ImageOps
from albumentations import BasicTransform
import numpy as np
import random
import torch

from PIL import Image

class AbstractPolicy(object):
    """ Randomly choose one of the best Sub-policies on a specific dataset.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = self._get_policies(fillcolor)

    @abstractstaticmethod
    def _get_policies(fillcolor):
        """Get the list of the best subpolicies computed with AutoAugment method for the specified dataset

        Return:
            List of the best subpolicies for the specified dataset"""

        raise NotImplementedError

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment {}".format(self.__class__.__name__)


class AbstractBackwardPolicy(BasicTransform):
    """ Randomly choose one of the best Sub-policies on a specific dataset for a couple image and mask.
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        super(AbstractBackwardPolicy, self).__init__(always_apply=True)
        self.policies = self._get_policies(fillcolor)
        self.backward_policies = []

    @abstractstaticmethod
    def _get_policies(fillcolor):
        """Get the list of the best subpolicies computed with AutoAugment method for the specified dataset

        Return:
            List of the best subpolicies for the specified dataset"""

        raise NotImplementedError

    @property
    def targets(self):
        return {'image': self.apply,
                'mask': self.apply_to_mask}

    def apply(self, img, **params):
        policy_idx = random.randint(0, len(self.policies) - 1)
        policy = self.policies[policy_idx]

        img = Image.fromarray((img * 255 / 2 + 128).astype(np.uint8))
        img = policy(img)
        img = np.array(img) / 255 - 128
        self.backward_policies = policy.applied_backward + self.policies

        return img

    def apply_to_mask(self, mask, **params):
        return mask

    def apply_backward(self, mask, **params):
        for op in self.backward_policies:
            if op is not None:
                mask = op(mask)

        return mask

    def __repr__(self):
        return "AutoAugment {}".format(self.__class__.__name__)


class CIFAR10Policy(AbstractPolicy):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10."""

    def __init__(self, **kwargs):
        super(CIFAR10Policy, self).__init__(**kwargs)

    @staticmethod
    def _get_policies(fillcolor):

        policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

        return policies


class ImageNetPolicy(AbstractPolicy):
    """ Randomly choose one of the best 20 Sub-policies on ImageNet."""

    def __init__(self, **kwargs):
        super(ImageNetPolicy, self).__init__(**kwargs)

    @staticmethod
    def _get_policies(fillcolor):

        policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

        ]

        return policies


class ImageNetBackwardPolicy(AbstractBackwardPolicy):
    """ Randomly choose one of the best 20 Sub-policies on ImageNet."""

    def __init__(self, **kwargs):
        super(ImageNetBackwardPolicy, self).__init__(**kwargs)

    @staticmethod
    def _get_policies(fillcolor):

        policies = [
            # SubBackwardPolicy(0.4, "posterize", 8, 0.6, "rotate90", 9),
            SubBackwardPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5),
            SubBackwardPolicy(0.8, "equalize", 8, 0.6, "equalize", 3),
            SubBackwardPolicy(0.6, "posterize", 7, 0.6, "posterize", 6),
            SubBackwardPolicy(0.4, "equalize", 7, 0.2, "solarize", 4),

            # SubBackwardPolicy(0.4, "equalize", 4, 0.8, "rotate90", 8),
            SubBackwardPolicy(0.6, "solarize", 3, 0.6, "equalize", 7),
            SubBackwardPolicy(0.8, "posterize", 5, 1.0, "equalize", 2),
            # SubBackwardPolicy(0.2, "rotate90", 3, 0.6, "solarize", 8),
            SubBackwardPolicy(0.6, "equalize", 8, 0.4, "posterize", 6),

            # SubBackwardPolicy(0.8, "rotate90", 8, 0.4, "color", 0),
            # SubBackwardPolicy(0.4, "rotate90", 9, 0.6, "equalize", 2),
            SubBackwardPolicy(0.0, "equalize", 7, 0.8, "equalize", 8),
            SubBackwardPolicy(0.6, "invert", 4, 1.0, "equalize", 8),
            SubBackwardPolicy(0.6, "color", 4, 1.0, "contrast", 8),

            # SubBackwardPolicy(0.8, "rotate90", 8, 1.0, "color", 2),
            SubBackwardPolicy(0.8, "color", 8, 0.8, "solarize", 7),
            SubBackwardPolicy(0.4, "sharpness", 7, 0.6, "invert", 8),
            SubBackwardPolicy(0.4, "color", 0, 0.6, "equalize", 3),
        ]

        return policies


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # func = {
        #     "shearX": lambda img, magnitude: img.transform(
        #         img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        #         Image.BICUBIC, fillcolor=fillcolor),
        #     "shearY": lambda img, magnitude: img.transform(
        #         img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        #         Image.BICUBIC, fillcolor=fillcolor),
        #     "translateX": lambda img, magnitude: img.transform(
        #         img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        #         fillcolor=fillcolor),
        #     "translateY": lambda img, magnitude: img.transform(
        #         img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        #         fillcolor=fillcolor),
        #     "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
        #     # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
        #     "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
        #     "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
        #     "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
        #     "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
        #         1 + magnitude * random.choice([-1, 1])),
        #     "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
        #         1 + magnitude * random.choice([-1, 1])),
        #     "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
        #         1 + magnitude * random.choice([-1, 1])),
        #     "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
        #     "equalize": lambda img, magnitude: ImageOps.equalize(img),
        #     "invert": lambda img, magnitude: ImageOps.invert(img)
        # }

        func = {
            "shearX": operations['ShearX'],
            "shearY": operations['ShearY'],
            "translateX": operations['TranslateX'],
            "translateY": operations['TranslateY'],
            "rotate": operations['Rotate'],
            "color": operations['Color'],
            "posterize": operations['Posterize'],
            "solarize": operations['Solarize'],
            "contrast": operations['Contrast'],
            "sharpness": operations['Sharpness'],
            "brightness": operations['Brightness'],
            "autocontrast": operations['AutoContrast'],
            "equalize": operations['Equalize'],
            "invert": operations['Invert']
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class SubBackwardPolicy(object):
    """Subpolicy with a backward relationship to revert the action (useful for segmentation).

    Two operations must be specified:
    - an operation applied to the image of the transformed branch
    - an operation on the mask of the not transformed branch"""

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2):
        ranges = {
            "rotate90": np.array([0, 1, 2, 3]),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "rotate90": [lambda img, magnitude: np.rot90(img, k=magnitude, axes=(0, 1)),
                         lambda mask, magnitude: torch.rot90(mask, k=magnitude, dims=(1, 2))],
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": [lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])), None],
            "posterize": [lambda img, magnitude: ImageOps.posterize(img, magnitude), None],
            "solarize": [lambda img, magnitude: ImageOps.solarize(img, magnitude), None],
            "contrast": [lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])), None],
            "sharpness": [lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])), None],
            "brightness": [lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])), None],
            "autocontrast": [lambda img, magnitude: ImageOps.autocontrast(img), None],
            "equalize": [lambda img, magnitude: ImageOps.equalize(img), None],
            "invert": [lambda img, magnitude: ImageOps.invert(img), None],
        }

        # func = {
        #     "rotate90": [lambda img, magnitude: np.rot90(img, k=magnitude, axes=(0, 1)),
        #                  lambda mask, magnitude: torch.rot90(mask, k=magnitude, dims=(1, 2))],
        #     # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
        #     "color": [operations['Color'], None],
        #     "posterize": [operations['Posterize'], None],
        #     "solarize": [operations['Solarize'], None],
        #     "contrast": [operations['Contrast'], None],
        #     "sharpness": [operations['Sharpness'], None],
        #     "brightness": [operations['Brightness'], None],
        #     "autocontrast": [operations['AutoContrast'], None],
        #     "equalize": [operations['Equalize'], None],
        #     "invert": [operations['Invert'], None],
        # }

        self.p1 = p1
        self.operation1 = func[operation1][0]
        self.backward_operation1 = func[operation1][1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2][0]
        self.backward_operation2 = func[operation2][1]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

        self.applied_backward = []

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
            self.applied_backward.append(self.backward_operation1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
            self.applied_backward.append(self.backward_operation2)
        return img
