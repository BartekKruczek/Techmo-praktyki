import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from dataloader import DataLoaderHandler
from dataloader_LFCC import DataLoaderHandlerLFCC
from dataloader_SPEC import DataLoaderHandlerSPEC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

class UtilsHandler:
    def __init__(self, data_path: str, classification: str, feature: int) -> None:
        """
        Feature: 1 -> mel spectrogram, 2 -> LFCC
        """
        self.data_path = data_path
        self.classification = classification
        self.feature = feature

    def __repr__(self) -> str:
        return f"Klasa do obsługi pomocniczych funkcji dla danych z folderu: {self.data_path}"
    
    def czech_language_pd(self) -> pd.DataFrame:
        data = []
        for root, dirs, files in os.walk(os.path.join(self.data_path, 'Czech_SLI')):
            for dir in dirs:
                if dir == "Healthy":
                    for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                        for subdir in subdirs:
                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                for file in file_files:
                                    if file.endswith(".wav"):
                                        file_path = os.path.join(file_root, file)
                                        if self.classification == "multi":
                                            data.append((file_path, "Czech", "healthy", "child", "healthy_child"))
                                        else:
                                            data.append((file_path, "Czech", "healthy", "child", "healthy_child"))
                elif dir == "Patients":
                    for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                        for subdir in subdirs:
                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                for file in file_files:
                                    if file.endswith(".wav"):
                                        file_path = os.path.join(file_root, file)
                                        if self.classification == "multi":
                                            data.append((file_path, "Czech", "SLI", "child", "SLI_child"))
                                        else:
                                            data.append((file_path, "Czech", "pathological", "child", "pathological_child"))
        return pd.DataFrame(data, columns=['file_path', 'language', 'healthy_status', 'gender', 'healthy_status_gender'])

    def english_language_pd(self) -> pd.DataFrame:
        data = []
        for root, dirs1, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria')):
            for dir1 in dirs1:
                if dir1 == "F":
                    for root, dirs2, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1)):
                        for dir2 in dirs2:
                            for root, dirs3, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1, dir2)):
                                for dir3 in dirs3:
                                    if dir3 == "Session1" or dir3 == "Session2" or dir3 == "Session3":
                                        for root, dirs4, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1, dir2, dir3)):
                                            for dir4 in dirs4:
                                                if dir4 == "wav_arrayMic" or dir4 == "wav_headMic":
                                                    for root, dirs5, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1, dir2, dir3, dir4)):
                                                        for file in files:
                                                            if file.endswith(".wav"):
                                                                file_path = os.path.join(root, file)
                                                                if self.classification == "multi":
                                                                    data.append((file_path, "English", "Dysarthria", "female", "DYSARTHRIA_female"))
                                                                else:
                                                                    data.append((file_path, "English", "pathological", "female", "pathological_female"))
                elif dir1 == "M":
                    for root, dirs2, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1)):
                        for dir2 in dirs2:
                            for root, dirs3, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1, dir2)):
                                for dir3 in dirs3:
                                    if dir3 == "Session1" or dir3 == "Session2" or dir3 == "Session3" or dir3 == "Session2_3":
                                        for root, dirs4, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1, dir2, dir3)):
                                            for dir4 in dirs4:
                                                if dir4 == "wav_arrayMic" or dir4 == "wav_headMic":
                                                    for root, dirs5, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria', dir1, dir2, dir3, dir4)):
                                                        for file in files:
                                                            if file.endswith(".wav"):
                                                                file_path = os.path.join(root, file)
                                                                if self.classification == "multi":
                                                                    data.append((file_path, "English", "Dysarthria", "male", "DYSARTHRIA_male"))
                                                                else:
                                                                    data.append((file_path, "English", "pathological", "male", "pathological_male"))
        return pd.DataFrame(data, columns=['file_path', 'language', 'healthy_status', 'gender', 'healthy_status_gender'])

    def german_language_pd(self) -> pd.DataFrame:
        data = []
        for root, dirs1, files in os.walk(os.path.join(self.data_path, 'SVD')):
            for dir1 in dirs1:
                if dir1 == "Cyste":
                    for root, dirs2, files in os.walk(os.path.join(self.data_path, 'SVD', dir1)):
                        for dir2 in dirs2:
                            if dir2 == "Healthy":
                                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                    for subdir in subdirs:
                                        if subdir == "F":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "healthy", "female", "healthy_female"))
                                                        else:
                                                            data.append((file_path, "German", "healthy", "female", "healthy_female"))
                                        elif subdir == "M":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "healthy", "male", "healthy_male"))
                                                        else:
                                                            data.append((file_path, "German", "healthy", "male", "healthy_male"))
                            elif dir2 == "Pathological":
                                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                    for subdir in subdirs:
                                        if subdir == "F":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "Cyste", "male", "CYSTE_male"))
                                                        else:
                                                            data.append((file_path, "German", "pathological", "male", "pathological_male"))
                                        elif subdir == "M":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "Cyste", "male"))
                                                        else:
                                                            data.append((file_path, "German", "pathological", "male"))
                elif dir1 == "Laryngitis":
                    for root, dirs2, files in os.walk(os.path.join(self.data_path, 'SVD', dir1)):
                        for dir2 in dirs2:
                            if dir2 == "Healthy":
                                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                    for subdir in subdirs:
                                        if subdir == "F":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "healthy", "female", "healthy_female"))
                                                        else:
                                                            data.append((file_path, "German", "healthy", "female", "healthy_female"))
                                        elif subdir == "M":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "healthy", "male", "healthy_male"))
                                                        else:
                                                            data.append((file_path, "German", "healthy", "male", "healthy_male"))
                            elif dir2 == "Pathological":
                                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                    for subdir in subdirs:
                                        if subdir == "F":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "Laryngitis", "male", "LARYNGITIS_female"))
                                                        else:
                                                            data.append((file_path, "German", "pathological", "male", "pathological_female"))
                                        elif subdir == "M":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "Laryngitis", "male", "LARYNGITIS_male"))
                                                        else:
                                                            data.append((file_path, "German", "pathological", "male", "pathological_male"))
                elif dir1 == "Phonasthenie":
                    for root, dirs2, files in os.walk(os.path.join(self.data_path, 'SVD', dir1)):
                        for dir2 in dirs2:
                            if dir2 == "Healthy":
                                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                    for subdir in subdirs:
                                        if subdir == "F":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "healthy", "female", "healthy_female"))
                                                        else:
                                                            data.append((file_path, "German", "healthy", "female", "healthy_female"))
                                        elif subdir == "M":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "healthy", "male", "healthy_male"))
                                                        else:
                                                            data.append((file_path, "German", "healthy", "male", "healthy_male"))
                            elif dir2 == "Pathological":
                                for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                    for subdir in subdirs:
                                        if subdir == "F":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "Phonasthenie", "female", "PHONASTHENIE_female"))
                                                        else:
                                                            data.append((file_path, "German", "pathological", "female", "pathological_female"))
                                        elif subdir == "M":
                                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                                for file in file_files:
                                                    if file.endswith(".wav"):
                                                        file_path = os.path.join(file_root, file)
                                                        if self.classification == "multi":
                                                            data.append((file_path, "German", "Phonasthenie", "male", "PHONASTHENIE_male"))
                                                        else:
                                                            data.append((file_path, "German", "pathological", "male", "pathological_male"))
        return pd.DataFrame(data, columns=['file_path', 'language', 'healthy_status', 'gender', 'healthy_status_gender'])

    def combined_language_pd(self) -> pd.DataFrame:
        czech_df = self.czech_language_pd()
        english_df = self.english_language_pd()
        german_df = self.german_language_pd()
        common_voice_df = self.common_voice()
        
        combined_df = pd.concat([czech_df, english_df, german_df, common_voice_df], ignore_index=True)
        # combined_df = pd.concat([czech_df, english_df, german_df], ignore_index=True)
        return combined_df
    
    def excel_creator(self, data_frame: pd.DataFrame) -> None:
        data_frame.to_excel("combined_languages.xlsx", index=False)

    def dataframe_from_excel(self, excel_file: str) -> pd.DataFrame:
        return pd.read_excel(excel_file)
    
    def common_voice(self) -> pd.DataFrame:
        filepaths = ["Database/Common_voice/cv-invalid.csv", "Database/Common_voice/cv-other-dev.csv", "Database/Common_voice/cv-other-test.csv",
                     "Database/Common_voice/cv-other-train.csv", "Database/Common_voice/cv-valid-dev.csv", "Database/Common_voice/cv-valid-test.csv",
                     "Database/Common_voice/cv-valid-train.csv"]
        
        dataframes = [pd.read_csv(file, sep = ",") for file in filepaths]

        combined_df = pd.concat(dataframes, ignore_index = True)

        # drop NaN from gender column
        combined_df = combined_df.dropna(subset = ['gender'])

        # get random 10k male and 10k female samples
        male_df = combined_df[combined_df['gender'] == 'male'].sample(n=3500, random_state=42)
        female_df = combined_df[combined_df['gender'] == 'female'].sample(n=3500, random_state=42)
        combined_df = pd.concat([male_df, female_df], ignore_index = True)

        # drop 'text', 'up_votes', 'down_votes', 'age', 'accent', 'duration' columns
        combined_df = combined_df.drop(columns = ['text', 'up_votes', 'down_votes', 'age', 'accent', 'duration'])

        # rename filename to file_path, add prefix
        combined_df = combined_df.rename(columns = {'filename': 'file_path'})
        combined_df['file_path'] = combined_df['file_path'].apply(lambda x: os.path.join('Database/Common_voice/', x))

        # add healthy_status column, every where healty
        combined_df['healthy_status'] = 'healthy'

        # add language column, every where English
        combined_df['language'] = 'English'

        # create new column 'healthy_status_gender', fill with 'healthy_female' or 'healthy_male'
        combined_df['healthy_status_gender'] = combined_df.apply(lambda x: f"{x['healthy_status']}_{x['gender']}", axis=1)

        return combined_df

    def split_dataset(self, dataframe: pd.DataFrame, device: torch.device):
        dataset_size = len(dataframe)
        indices = list(range(dataset_size))
        split1 = int(np.floor(0.6 * dataset_size))
        split2 = int(np.floor(0.8 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        if self.feature == 1:
            train_loader = DataLoader(DataLoaderHandler(dataframe, device, augmentation=True, classification=self.classification), batch_size=32, sampler=train_sampler, collate_fn=self.padd_input)
            val_loader = DataLoader(DataLoaderHandler(dataframe, device, augmentation=False, classification=self.classification), batch_size=32, sampler=val_sampler, collate_fn=self.padd_input)
            test_loader = DataLoader(DataLoaderHandler(dataframe, device, augmentation=False, classification=self.classification), batch_size=32, sampler=test_sampler, collate_fn=self.padd_input)

            return train_loader, val_loader, test_loader
        elif self.feature == 2:
            train_loader = DataLoader(DataLoaderHandlerLFCC(dataframe, device, augmentation=True, classification=self.classification), batch_size=32, sampler=train_sampler, collate_fn=self.padd_input)
            val_loader = DataLoader(DataLoaderHandlerLFCC(dataframe, device, augmentation=False, classification=self.classification), batch_size=32, sampler=val_sampler, collate_fn=self.padd_input)
            test_loader = DataLoader(DataLoaderHandlerLFCC(dataframe, device, augmentation=False, classification=self.classification), batch_size=32, sampler=test_sampler, collate_fn=self.padd_input)

            return train_loader, val_loader, test_loader
        elif self.feature == 3:
            train_loader = DataLoader(DataLoaderHandlerSPEC(dataframe, device, augmentation=True, classification=self.classification), batch_size=32, sampler=train_sampler, collate_fn=self.padd_input)
            val_loader = DataLoader(DataLoaderHandlerSPEC(dataframe, device, augmentation=False, classification=self.classification), batch_size=32, sampler=val_sampler, collate_fn=self.padd_input)
            test_loader = DataLoader(DataLoaderHandlerSPEC(dataframe, device, augmentation=False, classification=self.classification), batch_size=32, sampler=test_sampler, collate_fn=self.padd_input)

            return train_loader, val_loader, test_loader
    
    def split_dataset_stratified(self, dataframe: pd.DataFrame, device: torch.device, sample_size: int):
        stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        stratified_columns = ['healthy_status', 'language', 'gender']
        
        for sample_index, _ in stratified_split.split(dataframe, dataframe[stratified_columns]):
            sampled_df = dataframe.iloc[sample_index]
        
        X = sampled_df['file_path']
        y = sampled_df[['healthy_status', 'language', 'gender']]

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in stratified_split.split(X, y):
            train_df, test_df = sampled_df.iloc[train_index], sampled_df.iloc[test_index]

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for train_index, val_index in stratified_split.split(train_df['file_path'], train_df[['healthy_status', 'language', 'gender']]):
            train_df, val_df = train_df.iloc[train_index], train_df.iloc[val_index]

        if self.feature == 1:
            train_loader = DataLoader(DataLoaderHandler(train_df, device, augmentation=True, classification=self.classification), batch_size=32, collate_fn=self.padd_input)
            val_loader = DataLoader(DataLoaderHandler(val_df, device, augmentation=False, classification=self.classification), batch_size=32, collate_fn=self.padd_input)
            test_loader = DataLoader(DataLoaderHandler(test_df, device, augmentation=False, classification=self.classification), batch_size=32, collate_fn=self.padd_input)

            return train_loader, val_loader, test_loader
        elif self.feature == 2:
            train_loader = DataLoader(DataLoaderHandlerLFCC(train_df, device, augmentation=True, classification=self.classification), batch_size=32, collate_fn=self.padd_input)
            val_loader = DataLoader(DataLoaderHandlerLFCC(val_df, device, augmentation=False, classification=self.classification), batch_size=32, collate_fn=self.padd_input)
            test_loader = DataLoader(DataLoaderHandlerLFCC(test_df, device, augmentation=False, classification=self.classification), batch_size=32, collate_fn=self.padd_input)

            return train_loader, val_loader, test_loader
        elif self.feature == 3:
            train_loader = DataLoader(DataLoaderHandlerSPEC(train_df, device, augmentation=True, classification=self.classification), batch_size=32, collate_fn=self.padd_input)
            val_loader = DataLoader(DataLoaderHandlerSPEC(val_df, device, augmentation=False, classification=self.classification), batch_size=32, collate_fn=self.padd_input)
            test_loader = DataLoader(DataLoaderHandlerSPEC(test_df, device, augmentation=False, classification=self.classification), batch_size=32, collate_fn=self.padd_input)

            return train_loader, val_loader, test_loader
    
    def train_test_split(self, dataframe: pd.DataFrame):
        train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)
        return train_df, test_df
    
    def padd_input(self, batch):
        inputs, labels, demographics = zip(*batch)

        max_len = max([x.shape[-1] for x in inputs])

        inputs_padded = [F.pad(x, (0, max_len - x.shape[-1]), 'constant', 0) for x in inputs]
        inputs_padded = torch.stack(inputs_padded)

        inputs_padded = inputs_padded.squeeze(1)
        labels = torch.tensor(labels)
        demographics = torch.tensor(demographics)

        return inputs_padded, labels, demographics
