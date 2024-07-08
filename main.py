import numpy as np
import torch

from data import DataHandler
from utils import UtilsHandler
from dataloader import DataLoaderHandler
from model import Model
from train import TrainHandler
from torch.utils.data import SubsetRandomSampler

def main():
    my_data = DataHandler("Database")
    my_utils = UtilsHandler("Database")

    # pre-processing section
    pre_processing = False
    if pre_processing:
        print(f"Total of gathered sample: {my_data.all_languages_counter()}")
        my_data.plot_statistics(my_data.all_languages_counter())
        print(my_utils.combined_language_pd().head())

    # excel creation section
    excel_creation = False
    if excel_creation:
        my_utils.excel_creator(my_utils.combined_language_pd())

    dataframe = my_utils.dataframe_from_excel("combined_languages.xlsx")

    # dataloader section
    data_loader = DataLoaderHandler(dataframe)
    print(data_loader.__len__())

    dataset_size = len(data_loader)
    indices = list(range(dataset_size))
    split1 = int(np.floor(0.6 * dataset_size))
    split2 = int(np.floor(0.8 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(data_loader, batch_size=32, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(data_loader, batch_size=32, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(data_loader, batch_size=32, sampler=test_sampler)

    print(f"Train set size: {len(train_loader)}")
    print(f"Validation set size: {len(val_loader)}")
    print(f"Test set size: {len(test_loader)}")

if __name__ == '__main__':
    main()