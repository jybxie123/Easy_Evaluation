from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur, MocoGaussianBlur  # <fix>
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from utils import *


class ContrastiveLearningDataset:
    def __init__(self, args):
        self.root_folder = args.data
        self.args = args

    # @staticmethod
    def get_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(self.args.brightness * s, self.args.contrast * s,
                                              self.args.saturation * s, self.args.hue * s)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        data_transforms = None
        if self.args.cl_model == 'SimCLR':
            # the strength of RandomResizedCrop augmentation as r = (1-b) + (1-a)
            # scale:aug —> {(0.95, 1):0.05, (0.7, 1):0.3, (0.4, 1):0.6, (0.08, 1):0.92,
            # (0.2, 0.6):1.2, (0.2, 0.3):1.5,(0.02, 0.03):1.95}
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(eval(self.args.ResizedCropScale)[0], eval(self.args.ResizedCropScale)[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor()])
        return data_transforms

    def get_train_dataset(self, name, n_views, data_setup, train_trans=True):
        train_datasets = {### MNIST series
                          # load from the existing Meta-set MNIST, training sample size: 5w
                          'mnist': lambda: MyMNIST(self.root_folder + 'mnist', 'train_data.npy', 'train_label.npy',
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans)),

                          # deprecated, torchvision MNIST, training sample size: 6w, RGB Image mode
                          'mnist_raw': lambda: MyMNISTRAW(self.root_folder, train=True,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    download=True),  # Don't worry, it will skip if it already exists.

                          'fashion_mnist': lambda: MyFashionMNIST(self.root_folder, train=True,
                                                   transform=ContrastiveLearningViewGenerator(
                                                       self.get_simclr_pipeline_transform(28),
                                                       n_views,
                                                       data_setup,
                                                       train_trans),
                                                   download=True),

                          'k_mnist': lambda: MyKMNIST(self.root_folder, train=True,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    download=True),
        }
        try:
            dataset_fn = train_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # validation dataset for evaluating the two model acc of each epoch
    def get_val_dataset(self, name, n_views, data_setup, train_trans=False):
        val_datasets = {### MNIST series
                        'mnist': lambda: MyMNIST(self.root_folder + 'MNIST', 'test_data.npy', 'test_label.npy',
                                     transform=ContrastiveLearningViewGenerator(
                                         self.get_simclr_pipeline_transform(28),
                                         n_views,
                                         data_setup,
                                         train_trans)
                                     ),

                        # <fix>, deprecated, torchvision MNIST, training sample size: 6w, RGB Image mode
                        'mnist_raw': lambda: MyMNISTRAW(self.root_folder, train=False,
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(28),
                                                            n_views,
                                                            data_setup,
                                                            train_trans),
                                                        download=True),

                        'fashion_mnist': lambda: MyFashionMNIST(self.root_folder, train=False,
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_simclr_pipeline_transform(28),
                                                                    n_views,
                                                                    data_setup,
                                                                    train_trans),
                                                                download=True),

                        'k_mnist': lambda: MyKMNIST(self.root_folder, train=False,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(28),
                                                        n_views,
                                                        data_setup,
                                                        train_trans),
                                                    download=True),
                        }

        try:
            dataset_fn = val_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # seed sets for synthesizing meta dataset, usually the original dataset itself
    def get_seed_dataset(self, root_folder, name):
        NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        te_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(*NORM)])

        seed_datasets = {## MNIST series, Single Channels
                        'mnist': lambda: MNIST_bg(root_folder + 'MNIST', 'test_data.npy', 'test_label.npy'),

                        'fashion_mnist': lambda: datasets.FashionMNIST(root_folder, train=False, download=True),

                        'k_mnist': lambda: datasets.KMNIST(root_folder, train=False, download=True),
        }

        try:
            dataset_fn = seed_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # meta sets for training a regression model (If just need a correlation value, don't need this step)
    def get_meta_dataset(self, root_folder, name, n_views, data_setup, train_trans=False):
        meta_datasets = {### MNIST series, Single Channels
                        'mnist': lambda: MyMNIST(root_folder, 'test_data.npy', 'test_label.npy',
                                                 transform=ContrastiveLearningViewGenerator(
                                                     self.get_simclr_pipeline_transform(28),
                                                     n_views,
                                                     data_setup,
                                                     train_trans)
                                                 ),
        }

        try:
            dataset_fn = meta_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()

    # actually it's unseen target test dataset for evaluating model performance predictions
    def get_test_dataset(self, root_folder, name, n_views, data_setup, train_trans=False):
        test_datasets = {'svhn': lambda: datasets.SVHN(root_folder + 'SVHN', split='test',
                                                       transform=transforms.Compose([
                                                           transforms.Resize(28),
                                                           # convert into the MNIST-type size
                                                           ContrastiveLearningViewGenerator(
                                                               self.get_simclr_pipeline_transform(28),
                                                               n_views,
                                                               data_setup,
                                                               train_trans)]),
                                                       download=True),

                         'usps': lambda: datasets.USPS(root_folder + 'USPS', train=False,
                                                       transform=transforms.Compose([
                                                           transforms.Grayscale(num_output_channels=3),
                                                           # convert into the MNIST-type channels
                                                           transforms.Resize(28),
                                                           # convert into the MNIST-type size
                                                           ContrastiveLearningViewGenerator(
                                                               self.get_simclr_pipeline_transform(28),
                                                               n_views,
                                                               data_setup,
                                                               train_trans)]),
                                                       download=True),
                         }

        try:
            dataset_fn = test_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()