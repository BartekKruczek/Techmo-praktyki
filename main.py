from data import DataHandler
from utils import UtilsHandler
from feature_extraction import FeatureExtractionHandler
from dataloader import DataLoaderHandler

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

if __name__ == '__main__':
    main()