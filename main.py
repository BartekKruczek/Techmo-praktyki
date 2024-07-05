from data import DataHandler
from utils import UtilsHandler
from feature_extraction import FeatureExtractionHandler
from dataloader import DataLoaderHandler
from model import Model
from train import TrainHandler

def main():
    my_data = DataHandler("Database")
    my_utils = UtilsHandler("Database")
    my_feature = FeatureExtractionHandler()

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

    # audio path iteration section, MFCC extraction and Excel update
    update_excel = False
    if update_excel:
        y_values, sr_values, mfcc_values = my_feature.get_MFCC()

    # dataframe from excel section
    dataframe = my_utils.dataframe_from_excel("updated_combined_languages.xlsx")

    # split data section, three sets: train, validation, test
    my_dataloader = DataLoaderHandler(dataframe = dataframe)
    X_train, X_test, y_train, y_test = my_dataloader.split_data()
    
    # debug
    print(f"X_train type: {type(X_train)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train type: {type(y_train)}")
    print(f"y_train shape: {y_train.shape}")

    # model training section
    model = Model()
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10

    train_model = False
    if train_model:
        train = TrainHandler(model = model, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, batch_size = batch_size, learning_rate = learning_rate, num_epochs = num_epochs)
        train.train()
    


if __name__ == '__main__':
    main()