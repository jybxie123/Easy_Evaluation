import numpy as np
from torchvision.transforms import transforms

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2, data_setup=None, train_trans=True):
        self.base_transform = base_transform
        self.n_views = n_views
        self.train_trans = train_trans

        self.train_transform = None
        self.test_transform = None

        # MNIST Setup
        if data_setup == "mnist":
            self.normalize = transforms.Normalize((0.5,), (0.5,))
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

    def __call__(self, x):
        if self.train_trans:  # train set
            return [self.base_transform(x) for _ in range(self.n_views-1)] + [self.train_transform(x)]
        else:                 # test set
            return [self.base_transform(x) for _ in range(self.n_views - 1)] + [self.test_transform(x)]
