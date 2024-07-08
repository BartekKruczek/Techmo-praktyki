import warnings
import torch

from data import DataHandler
from utils import UtilsHandler
from dataloader import DataLoaderHandler
from model import Model
from train import TrainHandler

def main():
    case = 2
    if case == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
        print(f"Device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")
    
    # ignore warnings
    warnings.filterwarnings("ignore")

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
    print(f"NaN values in dataframe: {dataframe.isnull().sum().sum()}")

    # dataloader section
    data_loader = DataLoaderHandler(dataframe, device)
    print(f"Len dataloader {data_loader.__len__()}")

    train_loader, val_loader, test_loader = my_utils.split_dataset(data_loader)

    print(f"Train set size: {len(train_loader)}")
    print(f"Validation set size: {len(val_loader)}")
    print(f"Test set size: {len(test_loader)}")

    # training section
    do_train = False
    if do_train:
        model = Model()
        train_handler = TrainHandler(model, train_loader, val_loader, test_loader, device)
        train_handler.train()

if __name__ == '__main__':
    main()