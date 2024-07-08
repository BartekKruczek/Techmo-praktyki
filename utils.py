import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import SubsetRandomSampler

class UtilsHandler:
    def __init__(self, data_path: str):
        self.data_path = data_path

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
                                        data.append((file_path, "Czech", "healthy"))
                elif dir == "Patients":
                    for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                        for subdir in subdirs:
                            for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                for file in file_files:
                                    if file.endswith(".wav"):
                                        file_path = os.path.join(file_root, file)
                                        data.append((file_path, "Czech", "pathological"))
        return pd.DataFrame(data, columns=['file_path', 'language', 'healthy_status'])

    def english_language_pd(self) -> pd.DataFrame:
        data = []
        for root, dirs1, files in os.walk(os.path.join(self.data_path, 'Torgo_dysarthria')):
            for dir1 in dirs1:
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
                                                            data.append((file_path, "English", "pathological"))
        return pd.DataFrame(data, columns=['file_path', 'language', 'healthy_status'])

    def german_language_pd(self) -> pd.DataFrame:
        data = []
        for root, dirs1, files in os.walk(os.path.join(self.data_path, 'SVD')):
            for dir1 in dirs1:
                for root, dirs2, files in os.walk(os.path.join(self.data_path, 'SVD', dir1)):
                    for dir2 in dirs2:
                        if dir2 == "Healthy":
                            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                for subdir in subdirs:
                                    for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                        for file in file_files:
                                            if file.endswith(".wav"):
                                                file_path = os.path.join(file_root, file)
                                                data.append((file_path, "German", "healthy"))
                        elif dir2 == "Pathological":
                            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir2)):
                                for subdir in subdirs:
                                    for file_root, _, file_files in os.walk(os.path.join(subroot, subdir)):
                                        for file in file_files:
                                            if file.endswith(".wav"):
                                                file_path = os.path.join(file_root, file)
                                                data.append((file_path, "German", "pathological"))
        return pd.DataFrame(data, columns=['file_path', 'language', 'healthy_status'])

    def combined_language_pd(self) -> pd.DataFrame:
        czech_df = self.czech_language_pd()
        english_df = self.english_language_pd()
        german_df = self.german_language_pd()
        
        combined_df = pd.concat([czech_df, english_df, german_df], ignore_index=True)
        return combined_df
    
    def excel_creator(self, data_frame: pd.DataFrame) -> None:
        data_frame.to_excel("combined_languages.xlsx", index=False)

    def dataframe_from_excel(self, excel_file: str) -> pd.DataFrame:
        return pd.read_excel(excel_file)
    
    def split_dataset(self, data_loader: torch.utils.data.DataLoader):
        dataset_size = len(data_loader)
        indices = list(range(dataset_size))
        split1 = int(np.floor(0.6 * dataset_size))
        split2 = int(np.floor(0.8 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(data_loader, batch_size=32, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(data_loader, batch_size=32, sampler=val_sampler)
        test_loader = torch.utils.data.DataLoader(data_loader, batch_size=32, sampler=test_sampler)

        return train_loader, val_loader, test_loader

