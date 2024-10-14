# Import necessary libraries to check versions
import numpy as np
import pandas as pd
import matplotlib
import plotly
import cv2
import pydicom
import pylibjpeg
import tqdm
import PIL
import joblib
import sklearn
import torch
import torchvision

# Check and return the version of each package
versions = {
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'matplotlib': matplotlib.__version__,
    'plotly': plotly.__version__,
    'opencv-python': cv2.__version__,
    'pydicom': pydicom.__version__,
    'pylibjpeg': pylibjpeg.__version__,
    'tqdm': tqdm.__version__,
    'Pillow': PIL.__version__,
    'joblib': joblib.__version__,
    'scikit-learn': sklearn.__version__,
    'torch': torch.__version__,
    'torchvision': torchvision.__version__
}

print(versions)