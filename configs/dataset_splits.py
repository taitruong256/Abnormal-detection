"""
Dataset configurations for open-set recognition experiments.
Simple known/unknown splits without predefined trials.

All images are resized to 28×28.
Users specify 'known' classes, 'unknown' classes are auto-computed.
"""

# Dataset basic info
DATASET_INFO = {
    # BloodMNIST: 8 classes total
    # Microscopic images of individual normal blood cells
    # 17,092 color images (28×28)
    'bloodmnist': {
        'n_classes': 8,
        'channels': 3,  # RGB
        'image_size': 28,
        'description': 'Microscopic images of individual normal blood cells',
        'total_images': 17092,
    },
    
    # OCTMNIST: 4 classes total
    # Retinal OCT images (109,309 grayscale images, 28×28)
    'octmnist': {
        'n_classes': 4,
        'channels': 1,  # Grayscale (converted to RGB)
        'image_size': 28,
        'description': 'Retinal OCT images',
        'total_images': 109309,
    },
    
    # DermaMNIST: 7 classes total
    # Dermatoscopic images from HAM10000 dataset
    # 10,015 color images (28×28)
    'dermamnist': {
        'n_classes': 7,
        'channels': 3,  # RGB
        'image_size': 28,
        'description': 'Dermatoscopic images from HAM10000 dataset',
        'total_images': 10015,
    },
    
    # TissueMNIST: 8 classes total
    # Human kidney cortex cells from BBBC051 dataset
    # 236,386 grayscale images (28×28)
    'tissuemnist': {
        'n_classes': 8,
        'channels': 1,  # Grayscale (converted to RGB)
        'image_size': 28,
        'description': 'Human kidney cortex cells from BBBC051 dataset',
        'total_images': 236386,
    },
}


def get_unknown_classes(dataset_name, known):
    """
    Auto-compute unknown classes from known classes.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'bloodmnist')
        known: List of known class indices
        
    Returns:
        list: List of unknown class indices
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_INFO.keys())}")
    
    n_classes = DATASET_INFO[dataset_name]['n_classes']
    unknown = list(set(range(n_classes)) - set(known))
    
    return unknown


def print_dataset_info(dataset_name, known=None):
    """Print dataset information."""
    dataset_name = dataset_name.lower()
    
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_INFO.keys())}")
    
    info = DATASET_INFO[dataset_name]
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Description: {info['description']}")
    print(f"Total classes: {info['n_classes']}")
    print(f"Channels: {info['channels']} ({'Grayscale' if info['channels'] == 1 else 'RGB'})")
    print(f"Image size: {info['image_size']}×{info['image_size']}")
    print(f"Total images: {info['total_images']:,}")
    
    if known is not None:
        unknown = get_unknown_classes(dataset_name, known)
        print(f"\nKnown classes: {known} ({len(known)} classes)")
        print(f"Unknown classes: {unknown} ({len(unknown)} classes)")
    
    print(f"{'='*60}\n")


# Background dataset configuration
BACKGROUND_CONFIG = {
    'name': '300k_random_images',
    'num_samples': 300000,
    'description': 'Background samples to discriminate unknown classes from open space',
    'image_size': 28,
}


if __name__ == '__main__':
    """Test dataset configurations."""
    print("MedMNIST Dataset Information for Open-Set Recognition")
    print("="*60)
    
    # Example: BloodMNIST with 5 known classes
    print("\nExample 1: BloodMNIST with 5 known classes [0,1,2,3,4]")
    print_dataset_info('bloodmnist', known=[0, 1, 2, 3, 4])
    
    # Example: OCTMNIST with 2 known classes
    print("\nExample 2: OCTMNIST with 2 known classes [0,1]")
    print_dataset_info('octmnist', known=[0, 1])
    
    # Example: DermaMNIST with 4 known classes
    print("\nExample 3: DermaMNIST with 4 known classes [0,1,2,3]")
    print_dataset_info('dermamnist', known=[0, 1, 2, 3])
    
    # Example: TissueMNIST with 5 known classes
    print("\nExample 4: TissueMNIST with 5 known classes [0,1,2,3,4]")
    print_dataset_info('tissuemnist', known=[0, 1, 2, 3, 4])
