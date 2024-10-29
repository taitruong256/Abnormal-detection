import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
