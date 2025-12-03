"""
Dataset definitions for open-set recognition experiments.
Includes MedMNIST datasets and TinyImageNet for open-set evaluation.

Datasets: BloodMNIST (8 classes), OCTMNIST (4 classes), DermaMNIST (7 classes), TissueMNIST (8 classes)
Image size: 28×28
Background: 300k Random Images for open-space discrimination
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import medmnist
from medmnist import INFO, PathMNIST, DermaMNIST, OCTMNIST, TissueMNIST, BloodMNIST
from PIL import Image

# Add configs to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs'))
try:
    from dataset_splits import get_unknown_classes, DATASET_INFO
except ImportError:
    DATASET_INFO = None
    get_unknown_classes = None


class FilteredDataset(Dataset):
    """
    Helper class to filter and remap labels for open-set recognition.
    """
    def __init__(self, dataset, mask, known_classes):
        """
        Args:
            dataset: Base dataset
            mask: Boolean mask for filtering
            known_classes: List of known class labels
        """
        self.dataset = dataset
        self.indices = np.where(mask)[0]
        self.target_map = {label: idx for idx, label in enumerate(known_classes)}
    
    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        return img, self.target_map[int(label)]
    
    def __len__(self):
        return len(self.indices)


class Random300K_Images(Dataset):
    """
    TinyImageNet 300K images for open-set evaluation.
    """
    def __init__(self, file_path, transform=None, extendable=0):
        """
        Args:
            file_path: Path to .npy file containing 300k images
            transform: Optional transform
            extendable: Number of times to repeat data
        """
        self.transform = transform
        self.extendable = extendable
        self.offset = 0
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = np.load(file_path)
        if extendable > 0:
            self.data = np.repeat(self.data, extendable + 1, axis=0)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Label -1 indicates unknown/open-set
        return img, -1


class MedMNIST:
    """
    MedMNIST dataset wrapper for open-set recognition.
    
    Supports datasets:
    - BloodMNIST: 8 classes, 17,092 color images (28×28)
    - OCTMNIST: 4 classes, 109,309 grayscale images (28×28)
    - DermaMNIST: 7 classes, 10,015 color images (28×28)
    - TissueMNIST: 8 classes, 236,386 grayscale images (28×28)
    """
    
    def __init__(self, known, unknown=None, dataroot='./data', use_gpu=True, 
                 num_workers=4, batch_size=128, patch_size=28, gray_scale=False,
                 dataset_name='bloodmnist'):
        """
        Args:
            known: List of known class indices for training
            unknown: List of unknown class indices (optional, auto-computed from known if None)
            dataroot: Root directory for datasets
            use_gpu: Whether to use GPU (pin_memory)
            num_workers: Number of workers for data loading
            batch_size: Batch size for data loaders
            patch_size: Image size (default: 28)
            gray_scale: Whether to use grayscale (default: False, convert to RGB)
            dataset_name: MedMNIST dataset name (bloodmnist, octmnist, dermamnist, tissuemnist)
        """
        self.dataset_name = dataset_name.lower()
        self.gray_scale = gray_scale
        self.patch_size = patch_size
        
        # Get dataset info from MedMNIST
        self.info = INFO[self.dataset_name]
        self.task = self.info['task']
        self.n_channels = self.info['n_channels']
        self.n_classes = len(self.info['label'])
        
        # Create class_to_idx mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.info['label'].keys())}
        
        # Get known and unknown classes
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
        
        # Auto-compute unknown classes if not specified
        if unknown is not None:
            self.unknown = unknown
        else:
            self.unknown = list(set(range(self.n_classes)) - set(self.known))
        
        self.num_classes = len(self.known)
        
        print(f"\n{'='*60}")
        print(f"MedMNIST Dataset: {self.dataset_name.upper()}")
        print(f"{'='*60}")
        print(f"Task: {self.task}")
        print(f"Channels: {self.n_channels}")
        print(f"Total classes: {self.n_classes}")
        print(f"Known classes (closed-set): {self.known}")
        print(f"Unknown classes (open-set): {self.unknown}")
        print(f"{'='*60}\n")
        
        # Get transforms
        self.train_transforms, self.val_transforms = self.__get_transforms(patch_size, gray_scale)
        
        # Load datasets
        self.trainset, self.valset, self.outset = self.get_dataset(dataroot)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.out_loader = self.get_dataset_loader(
            batch_size, num_workers, use_gpu
        )
    
    def __get_transforms(self, patch_size, gray_scale=False):
        """
        Get train and test transforms with data augmentation.
        
        Args:
            patch_size: Image size (default: 28)
            gray_scale: Whether to use grayscale or convert to RGB
            
        Returns:
            train_transforms, val_transforms
        """
        # Data augmentation for training
        if gray_scale:
            train_transforms = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.Grayscale(3),
                transforms.RandomCrop(patch_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            val_transforms = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.Lambda(lambda x: x if x.mode == 'RGB' else x.convert('RGB')),
                transforms.RandomCrop(patch_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            val_transforms = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.Lambda(lambda x: x if x.mode == 'RGB' else x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        return train_transforms, val_transforms
    
    def get_dataset(self, dataroot):
        """
        Load and filter datasets by known/unknown classes.
        
        Args:
            dataroot: Root directory for datasets
            
        Returns:
            trainset, valset, outset (filtered datasets)
        """
        # Get the dataset class
        dataset_mapping = {
            'pathmnist': PathMNIST,
            'dermamnist': DermaMNIST,
            'octmnist': OCTMNIST,
            'tissuemnist': TissueMNIST,
            'bloodmnist': BloodMNIST,
        }
        
        if self.dataset_name not in dataset_mapping:
            # Fallback to dynamic loading for other MedMNIST datasets
            DataClass = getattr(medmnist, self.info['python_class'])
        else:
            DataClass = dataset_mapping[self.dataset_name]
        
        # Load raw datasets
        train_dataset = DataClass(
            split='train',
            transform=self.train_transforms,
            download=True,
            root=dataroot
        )
        
        test_dataset = DataClass(
            split='test',
            transform=self.val_transforms,
            download=True,
            root=dataroot
        )
        
        # Print class distribution
        train_labels = train_dataset.labels.squeeze()
        print(f"{self.dataset_name.upper()} Training set class distribution: {np.bincount(train_labels)}")
        
        # Filter datasets by known and unknown classes
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)
        
        # Create filtered datasets
        trainset = FilteredDataset(train_dataset, train_mask, self.known)
        valset = FilteredDataset(test_dataset, known_test_mask, self.known)
        outset = FilteredDataset(test_dataset, unknown_test_mask, self.unknown)
        
        print(f'{self.dataset_name.upper()} Train samples: {len(trainset)}')
        print(f'{self.dataset_name.upper()} Test samples (known): {len(valset)}')
        print(f'{self.dataset_name.upper()} Test samples (unknown): {len(outset)}\n')
        
        return trainset, valset, outset
    
    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Create data loaders.
        
        Args:
            batch_size: Batch size for data loaders
            workers: Number of workers for data loading
            is_gpu: Whether to use GPU (pin_memory)
            
        Returns:
            train_loader, val_loader, out_loader
        """
        train_loader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=is_gpu,
            sampler=None
        )
        
        val_loader = DataLoader(
            self.valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=is_gpu
        )
        
        out_loader = DataLoader(
            self.outset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=is_gpu
        )
        
        return train_loader, val_loader, out_loader


class TinyImageNetOpenSet(Dataset):
    """
    TinyImageNet dataset for open-set evaluation.
    Loads 300k images from TinyImageNet to use as unknown/novel class data.
    """
    
    def __init__(self, root_dir, num_samples=300000, transform=None, download=True, patch_size=28):
        """
        Args:
            root_dir: Root directory to store TinyImageNet data
            num_samples: Number of images to use (default: 300k)
            transform: Optional transform to apply to images
            download: Whether to download the dataset if not present
            patch_size: Image size (default: 28)
        """
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.data_dir = os.path.join(root_dir, 'tiny-imagenet-200')
        
        # Default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        if download and not os.path.exists(self.data_dir):
            self._download_tinyimagenet()
        
        # Load image paths
        self.image_paths = self._load_image_paths()
        
        # Limit to num_samples
        if len(self.image_paths) > num_samples:
            indices = np.random.choice(len(self.image_paths), num_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
        
        print(f"\nTinyImageNet Open-Set Dataset:")
        print(f"  Total images loaded: {len(self.image_paths)}")
        print(f"  Image size: {patch_size}x{patch_size}")
        print(f"  Location: {self.data_dir}\n")
    
    def _download_tinyimagenet(self):
        """Download TinyImageNet dataset."""
        import urllib.request
        import zipfile
        
        print("Downloading TinyImageNet dataset (237MB)...")
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(self.root_dir, 'tiny-imagenet-200.zip')
        
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Download
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        print("Extracting TinyImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print("TinyImageNet download complete!")
    
    def _load_image_paths(self):
        """Load all image paths from TinyImageNet."""
        image_paths = []
        
        # Load from train directory
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.exists(train_dir):
            for class_dir in os.listdir(train_dir):
                class_path = os.path.join(train_dir, class_dir, 'images')
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith('.JPEG'):
                            image_paths.append(os.path.join(class_path, img_name))
        
        # Load from val directory
        val_dir = os.path.join(self.data_dir, 'val', 'images')
        if os.path.exists(val_dir):
            for img_name in os.listdir(val_dir):
                if img_name.endswith('.JPEG'):
                    image_paths.append(os.path.join(val_dir, img_name))
        
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: Always -1 (indicating unknown/open-set class)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Label -1 indicates unknown/open-set class
        return image, -1


def get_dataset(is_gpu, args):
    """
    Factory function to get the appropriate dataset.
    
    Args:
        is_gpu: Whether to use GPU (pin_memory)
        args: Arguments containing dataset configuration
            - dataset / dataset_name: Dataset name
            - known: List of known class indices
            - unknown: List of unknown class indices (optional)
            - dataroot: Root directory for datasets
            - batch_size: Batch size
            - workers / num_workers: Number of workers
            - patch_size: Image size (default: 28)
            - gray_scale: Whether to use grayscale (default: False)
        
    Returns:
        Dataset object with train_loader, val_loader, out_loader
    """
    # Get dataset name (try both 'dataset' and 'dataset_name')
    if hasattr(args, 'dataset_name'):
        dataset_name = args.dataset_name.lower()
    elif hasattr(args, 'dataset'):
        dataset_name = args.dataset.lower()
    else:
        raise ValueError("args must have 'dataset_name' or 'dataset' attribute")
    
    # Check if it's a MedMNIST dataset
    supported_datasets = ['bloodmnist', 'octmnist', 'dermamnist', 'tissuemnist']
    medmnist_datasets = [
        'pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
        'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
        'organamnist', 'organcmnist', 'organsmnist', 'chestmnist'
    ]
    
    if dataset_name in medmnist_datasets:
        print(f"Loading MedMNIST dataset: {dataset_name}")
        
        # Extract parameters
        # Known classes (required)
        if hasattr(args, 'known'):
            known = args.known
        else:
            # Default: use approximately 60% of classes as known
            n_classes = len(INFO[dataset_name]['label'])
            known = list(range(int(n_classes * 0.6)))
            print(f"Warning: 'known' not specified, using default {known}")
        
        # Unknown classes (optional, auto-computed from known)
        unknown = args.unknown if hasattr(args, 'unknown') else None
        
        dataroot = args.dataroot if hasattr(args, 'dataroot') else './data'
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 128
        num_workers = args.workers if hasattr(args, 'workers') else (args.num_workers if hasattr(args, 'num_workers') else 4)
        patch_size = args.patch_size if hasattr(args, 'patch_size') else 28
        gray_scale = args.gray_scale if hasattr(args, 'gray_scale') else False
        
        # Create dataset
        dataset = MedMNIST(
            known=known,
            unknown=unknown,
            dataroot=dataroot,
            use_gpu=is_gpu,
            num_workers=num_workers,
            batch_size=batch_size,
            patch_size=patch_size,
            gray_scale=gray_scale,
            dataset_name=dataset_name
        )
        
        return dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {medmnist_datasets}")

