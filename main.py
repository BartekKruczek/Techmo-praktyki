import warnings
import torch
import optuna

from data import DataHandler
from utils import UtilsHandler
from dataloader import DataLoaderHandler
from model import Model
from train import TrainHandler

def objective(trial):
    case = 2
    if case == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
        print(f"Device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")
    
    # ignore warnings
    warnings.filterwarnings("ignore")

    my_utils = UtilsHandler("Database")
    my_data = DataHandler("Database", my_utils)

    dataframe = my_utils.dataframe_from_excel("combined_languages.xlsx")
    print(f"NaN values in dataframe: {dataframe.isnull().sum().sum()}")
    print(f"Dataframe head: {dataframe.head()}")

    # data section
    create_excel: bool = True
    if create_excel:
        combined_df = my_utils.combined_language_pd()
        my_utils.excel_creator(combined_df)
        my_utils.common_voice()

    create_png: bool = True
    if create_png:
        languages = my_data.all_languages_counter()
        my_data.plot_statistics(languages)
        my_data.gender_statistic_png()
        my_data.audio_files_length_histogram(dataframe)

    # dataloader section
    data_loader = DataLoaderHandler(dataframe, device, augmentation = True)
    print(f"Len dataloader {data_loader.__len__()}")

    train_loader, val_loader, test_loader = my_utils.split_dataset(data_loader)

    print(f"Train set size: {len(train_loader)}")
    print(f"Validation set size: {len(val_loader)}")
    print(f"Test set size: {len(test_loader)}")

    # tuning hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # num_epochs = trial.suggest_int("num_epochs", 5, 50)
    step_size = trial.suggest_int("step_size", 1, 20)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-2, log=True)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True)

    num_epochs: int = 5

    do_train: bool = False
    if do_train:
        model = Model()
        train_handler = TrainHandler(model, train_loader, val_loader, test_loader, device, learning_rate, num_epochs, step_size, gamma, l1_lambda, l2_lambda)
        accuracy = train_handler.train()
        return accuracy
    else:
        return 0.0

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best parameters: {study.best_params}")
