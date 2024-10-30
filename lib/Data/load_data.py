import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lib.Data.datasets import BreastCancerDataset
import os
import kagglehub

def download_kaggle_dataset(dataset_name):
    data_path = kagglehub.dataset_download(dataset_name)
    print(f'Download completed at path: {data_path}.')
    return data_path 

def load_and_concatenate_datasets(dataset_paths):
    df_list = []
    for idx, dataset_path in enumerate(dataset_paths, start=1):
        csv_path = os.path.join(dataset_path, 'train.csv')
        df = pd.read_csv(csv_path)
        df['png_path'] = df.apply(
            lambda row: f"{dataset_path}/train_images/{row['patient_id']}/{row['image_id']}.png",
            axis=1
        )
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
    

def split_data(df, batch_size, input_shape):
    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_close_df      = train_df[train_df.cancer==0]
    train_open_df       = train_df[train_df.cancer==1]
    val_close_df        = val_df[val_df.cancer==0]
    val_open_df         = val_df[val_df.cancer==1]

    # Create subsets for open and close datasets (cancer and non-cancer)
    train_open_dataset  = BreastCancerDataset(train_open_df, input_shape)
    train_close_dataset = BreastCancerDataset(train_close_df, input_shape)
    train_dataset       = BreastCancerDataset(train_df, input_shape)
    val_open_dataset    = BreastCancerDataset(val_open_df, input_shape)
    val_close_dataset   = BreastCancerDataset(val_close_df, input_shape)
    val_dataset         = BreastCancerDataset(val_df, input_shape)

    # Create DataLoaders
    train_open_loader   = DataLoader(train_open_dataset,  batch_size=batch_size, shuffle=True)
    train_close_loader  = DataLoader(train_close_dataset, batch_size=batch_size, shuffle=True)
    train_loader        = DataLoader(train_dataset,       batch_size=batch_size, shuffle=True)
    val_open_loader     = DataLoader(val_open_dataset,    batch_size=batch_size, shuffle=False)
    val_close_loader    = DataLoader(val_close_dataset,   batch_size=batch_size, shuffle=False)
    val_loader          = DataLoader(val_dataset,         batch_size=batch_size, shuffle=False)
    
    return train_loader, train_open_loader, train_close_loader, val_loader, val_open_loader, val_close_loader
