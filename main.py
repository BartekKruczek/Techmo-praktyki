from data import DataHandler
from utils import UtilsHandler

def main():
    my_data = DataHandler("Database")
    my_utils = UtilsHandler("Database")

    # pre-processing section
    print(f"Total of gathered sample: {my_data.all_languages_counter()}")
    my_data.plot_statistics(my_data.all_languages_counter())
    print(my_utils.combined_language_pd().head())

    # excel creation section
    my_utils.excel_creator(my_utils.combined_language_pd())

if __name__ == '__main__':
    main()