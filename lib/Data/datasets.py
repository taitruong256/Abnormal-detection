import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets

class BreastCancerDataset(Dataset):
    def __init__(self, df, input_shape):
        self.df = df
        self.input_shape = input_shape
        
        self.transform = transforms.Compose([
            transforms.Resize((self.input_shape, self.input_shape)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        png_path = self.df.iloc[idx]['png_path']
        img_tensor = self.read_and_resize_image(png_path)

        label = self.df.iloc[idx]['cancer']
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

    def read_and_resize_image(self, png_path):
        img = Image.open(png_path).convert('L') 
        img_tensor = self.transform(img) 
        return img_tensor




class MNIST:
    """
    MNIST dataset featuring gray-scale 28x28 images of
    hand-written characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.MNIST.

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        class_to_idx (dict): Defines mapping from class names to integers.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.num_classes = 10
        self.gray_scale = args.gray_scale
        self.max_train_samples = args.max_train_samples
        self.max_test_samples = args.max_test_samples

        self.train_transforms, self.val_transforms = self.__get_transforms(args.patch_size)

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

        # Need to define the class dictionary by hand as the default
        # torchvision MNIST data loader does not provide class_to_idx
        self.class_to_idx = {'0': 0,
                             '1': 1,
                             '2': 2,
                             '3': 3,
                             '4': 4,
                             '5': 5,
                             '6': 6,
                             '7': 7,
                             '8': 8,
                             '9': 9}

    def __get_transforms(self, patch_size):
        # optionally scale the images and repeat them to three channels
        # important note: these transforms will only be called once during the
        # creation of the dataset and no longer in the incremental datasets that inherit.
        # Adding data augmentation here is thus the wrong place!

        if self.gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(size=(patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

        return train_transforms, val_transforms

    def get_dataset(self):
        """
        Uses torchvision.datasets.MNIST to load dataset.
        Downloads dataset if doesn't exist already.
        Applies max_train_samples and max_test_samples limits if specified.

        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        trainset = datasets.MNIST('datasets/MNIST/train/', train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        valset = datasets.MNIST('datasets/MNIST/test/', train=False, transform=self.val_transforms,
                                target_transform=None, download=True)

        # Apply max_train_samples limit
        if self.max_train_samples is not None:
            trainset = Subset(trainset, list(range(min(self.max_train_samples, len(trainset)))))

        # Apply max_test_samples limit (default: 20% of training set)
        if self.max_test_samples is not None:
            valset = Subset(valset, list(range(min(self.max_test_samples, len(valset)))))
        else:
            # If max_test_samples not specified, use 20% of training samples
            if self.max_train_samples is not None:
                test_size = max(1, int(self.max_train_samples * 0.2))
            else:
                test_size = len(valset)
            valset = Subset(valset, list(range(min(test_size, len(valset)))))

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.DataLoader: train_loader, val_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader