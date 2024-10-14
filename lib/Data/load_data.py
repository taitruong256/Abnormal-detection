import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lib.Data.datasets import BreastCancerDataset
import os

def load_data(batch_size):
    print("Current directory:", os.getcwd())
    # Load CSV
    df = pd.read_csv(r'E:\data\train.csv')
    print('Total dataset:', len(df))

    # Add 'png_path' to each dataframe
    df['png_path'] = df.apply(lambda row: os.path.join(r'E:\data\train_images', str(row['patient_id']), f"{row['image_id']}.png"), axis=1)

    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_close_df      = train_df[train_df.cancer==0]
    train_open_df       = train_df[train_df.cancer==1]
    val_close_df        = val_df[val_df.cancer==0]
    val_open_df         = val_df[val_df.cancer==1]

    # Create subsets for open and close datasets (cancer and non-cancer)
    train_open_dataset  = BreastCancerDataset(train_open_df)
    train_close_dataset = BreastCancerDataset(train_close_df)
    train_dataset       = BreastCancerDataset(train_df)
    val_open_dataset    = BreastCancerDataset(val_open_df)
    val_close_dataset   = BreastCancerDataset(val_close_df)
    val_dataset         = BreastCancerDataset(val_df)

    # Create datasets using the BreastCancerDataset class
    train_open_loader   = DataLoader(train_open_dataset,  batch_size=batch_size, shuffle=True)
    train_close_loader  = DataLoader(train_close_dataset, batch_size=batch_size, shuffle=True)
    train_loader        = DataLoader(train_dataset,       batch_size=batch_size, shuffle=True)
    val_open_loader     = DataLoader(val_open_dataset,    batch_size=batch_size, shuffle=False)
    val_close_loader    = DataLoader(val_close_dataset,   batch_size=batch_size, shuffle=False)
    val_loader          = DataLoader(val_dataset,         batch_size=batch_size, shuffle=False)

    # Create DataLoaders
    train_open_loader   = DataLoader(train_open_dataset,  batch_size=batch_size, shuffle=True)
    train_close_loader  = DataLoader(train_close_dataset, batch_size=batch_size, shuffle=True)
    train_loader        = DataLoader(train_dataset,       batch_size=batch_size, shuffle=True)
    val_open_loader     = DataLoader(val_open_dataset,    batch_size=batch_size, shuffle=False)
    val_close_loader    = DataLoader(val_close_dataset,   batch_size=batch_size, shuffle=False)
    val_loader          = DataLoader(val_dataset,         batch_size=batch_size, shuffle=False)
    
    return df, train_loader, train_open_loader, train_close_loader, val_loader, val_open_loader, val_close_loader
