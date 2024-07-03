from data import DataHandler

def main():
    my_data = DataHandler("Database")

    print(f"Total of gathered sample: {my_data.all_languages_counter()}")
    my_data.plot_statistics(my_data.all_languages_counter())

if __name__ == '__main__':
    main()