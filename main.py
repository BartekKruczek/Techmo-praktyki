import warnings
import torch
import optuna

from data import DataHandler
from utils import UtilsHandler
from dataloader import DataLoaderHandler
from model import Model
from model_RNN import ModelRNNHandler
from train import TrainHandler
from train_RNN import TrainHandlerRNN
from auto_features_extraction import AutoFeaturesExtraction
from dataloader_RNN import DataLoaderRNNHandler
from torchsummary import summary
from pytorch_model import PytorchModelHandler
from model_conv1d import ModelConv1D
from run_inz import run

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

    classif: str = "binary"
    feature = 3
    my_utils = UtilsHandler(data_path = "Database", classification = classif, feature = feature) 
    my_data = DataHandler(data_path = "Database", utils = my_utils, classification = classif)
    my_auto_features = AutoFeaturesExtraction()

    dataframe = my_utils.dataframe_from_excel("combined_languages.xlsx")
    # print(f"NaN values in dataframe: {dataframe.isnull().sum().sum()}")
    # print(f"Dataframe head: {dataframe.head()}")

    my_RNNHandler = DataLoaderRNNHandler(dataframe, device, augmentation = True, classification = classif)
    my_pytorchmode = PytorchModelHandler(dataframe, augmentation = True, classification = classif)

    # data section
    create_excel: bool = True
    if create_excel:
        combined_df = my_utils.combined_language_pd()
        my_utils.excel_creator(combined_df)
        my_utils.common_voice()

    create_png: bool = False
    if create_png:
        languages = my_data.all_languages_counter()
        my_data.plot_statistics(languages)
        my_data.gender_statistic_png()
        my_data.audio_files_length_histogram(dataframe)

    # dataloader section, case = 1 if features mel spectrogram, case = 2 if features from model
    case = 2
    if case == 1:
        data_loader = DataLoaderHandler(dataframe, device, augmentation=True, classification = classif)
        # print(f"Len dataloader {data_loader.__len__()}")
    elif case == 2:
        data_loader = DataLoaderRNNHandler(dataframe, device, augmentation=True, classification = classif)
        # print(f"Len dataloader RNN {data_loader.__len__()}")

    # print(data_loader.__getitem__(0))
    # print(f"Shape of 1-st element: {data_loader.__getitem__(0)[0].shape}")
    # print(f"Type: {type(data_loader.__getitem__(0))}")

    # X, Y = my_pytorchmode.get_X_Y_inz(dataframe)

    # podział na zbiory
    do_stratified: bool = False
    if do_stratified:
        train_loader, val_loader, test_loader = my_utils.split_dataset_stratified(dataframe, device, sample_size = 0.8)
    else:
        train_loader, val_loader, test_loader = my_utils.split_dataset(dataframe, device)

    # print(my_pytorchmode.extract_features())
    # print(my_pytorchmode.extract_features().shape)

    # my_RNNHandler.check_inputs_shape(train_loader)
    # my_RNNHandler.check_inputs_shape_after_padding(train_loader)

    # tuning hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # num_epochs = trial.suggest_int("num_epochs", 5, 50)
    step_size = trial.suggest_int("step_size", 1, 20)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-2, log=True)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True)

    num_epochs: int = 5

    do_train: bool = True
    case_train = 4

    if do_train and case_train == 1:
        model = Model(num_classes = 2)
        train_handler = TrainHandler(model, train_loader, val_loader, test_loader, device, learning_rate, num_epochs, step_size, gamma, l1_lambda, l2_lambda)
        accuracy = train_handler.train()
        return accuracy
    elif do_train and case_train == 2:
        model = ModelRNNHandler(input_size = 12, hidden_size = 64, num_layers = 5, num_classes = 2)
        # model = ModelConv1D(input_channels = 128, output_channels = 64, kernel_size = 3)
        # summary(model, (1, 12))
        train_handler = TrainHandlerRNN(model, train_loader, val_loader, test_loader, device, learning_rate, num_epochs, step_size, gamma, l1_lambda, l2_lambda)
        accuracy = train_handler.train()
        return accuracy
    elif do_train and case_train == 4:
        model = Model(num_classes = 2)
        train_handler = TrainHandler(model, train_loader, val_loader, test_loader, device, learning_rate, num_epochs, step_size, gamma, l1_lambda, l2_lambda)
        accuracy = train_handler.train()
        return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best parameters: {study.best_params}")