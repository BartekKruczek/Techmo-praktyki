from data import DataHandler
from utils import UtilsHandler
from feature_extraction import FeatureExtractionHandler

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

    # audio path iteration section
    my_feature.get_MFCC()

if __name__ == '__main__':
    main()