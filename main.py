import numpy as np

from data import DataHandler
from utils import UtilsHandler
from dataloader import DataLoaderHandler
from model import Model
from train import TrainHandler

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
    my_dataloader = DataLoaderHandler(dataframe = dataframe)
    for i in range(len(my_dataloader)):
        mfcc, label = my_dataloader[i]
        print(f"MFCC: {mfcc}, label: {label}")
    
if __name__ == '__main__':
    main()