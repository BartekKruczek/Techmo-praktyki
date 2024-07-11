import os
import matplotlib.pyplot as plt
import pandas as pd
import librosa

class DataHandler:
    def __init__(self, data_path: str, utils: classmethod) -> None:
        self.data_path = data_path
        self.utils = utils

    def __repr__(self) -> str:
        return f"Klasa do obsługi danych z folderu: {self.data_path}"
    
    def czech_language_counter(self) -> dict:
        """
        {language: {healthy: int, pathological: int}}
        """
        languages = {}
        
        # czech
        for root, dirs, files in os.walk(os.path.join(self.data_path, 'Czech_SLI')):
            for dir in dirs:
                if dir == "Healthy":
                    for root, dirs, files in os.walk(os.path.join(self.data_path, 'Czech_SLI', 'Healthy')):
                        for dir in dirs:
                            for root, dirs, files in os.walk(os.path.join(self.data_path, 'Czech_SLI', 'Healthy', dir)):
                                for file in files:
                                    if file.endswith(".wav"):
                                        # increment language counter
                                        if "Czech" not in languages:
                                            languages["Czech"] = {"healthy": 1, "pathological": 0}
                                        else:
                                            languages["Czech"]["healthy"] += 1
                elif dir == "Patients":
                    for root, dirs, files in os.walk(os.path.join(self.data_path, 'Czech_SLI', 'Patients')):
                        for dir in dirs:
                            for root, dirs, files in os.walk(os.path.join(self.data_path, 'Czech_SLI', 'Patients', dir)):
                                for file in files:
                                    if file.endswith(".wav"):
                                        # increment language counter
                                        if "Czech" not in languages:
                                            languages["Czech"] = {"healthy": 0, "pathological": 1}
                                        else:
                                            languages["Czech"]["pathological"] += 1

        return languages

    def english_language_counter(self) -> dict:
        languages = {}

        # english, Torgo only patients
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
                                                            # increment language counter
                                                            if "English" not in languages:
                                                                languages["English"] = {"healthy": 0, "pathological": 1}
                                                            else:
                                                                languages["English"]["pathological"] += 1
        return languages
    
    def german_language_counter(self) -> dict:
        languages = {}

        # german
        for root, dirs1, files in os.walk(os.path.join(self.data_path, 'SVD')):
            for dir1 in dirs1:
                for root, dirs2, files in os.walk(os.path.join(self.data_path, 'SVD', dir1)):
                    for dir2 in dirs2:
                        if dir2 == "Healthy":
                            for root, dirs3, files in os.walk(os.path.join(self.data_path, 'SVD', dir1, dir2)):
                                for dir3 in dirs3:  
                                    # print(dir3)
                                    for root, dirs4, files in os.walk(os.path.join(self.data_path, 'SVD', dir1, dir2, dir3)):
                                        for dir4 in dirs4:
                                            for root, dirs5, files in os.walk(os.path.join(self.data_path, 'SVD', dir1, dir2, dir3, dir4)):
                                                for file in files:
                                                    if file.endswith(".wav"):
                                                        # increment language counter
                                                        if "German" not in languages:
                                                            languages["German"] = {"healthy": 1, "pathological": 0}
                                                        else:
                                                            languages["German"]["healthy"] += 1
                        elif dir2 == "Pathological":
                            for root, dirs3, files in os.walk(os.path.join(self.data_path, 'SVD', dir1, dir2)):
                                for dir3 in dirs3:  
                                    # print(dir3)
                                    for root, dirs4, files in os.walk(os.path.join(self.data_path, 'SVD', dir1, dir2, dir3)):
                                        for dir4 in dirs4:
                                            for root, dirs5, files in os.walk(os.path.join(self.data_path, 'SVD', dir1, dir2, dir3, dir4)):
                                                for file in files:
                                                    if file.endswith(".wav"):
                                                        # increment language counter
                                                        if "German" not in languages:
                                                            languages["German"] = {"healthy": 0, "pathological": 1}
                                                        else:
                                                            languages["German"]["pathological"] += 1

        return languages
    
    def common_voice_language_counter(self) -> dict:
        languages = {}

        # common voice df
        df = self.utils.common_voice()

        # healthy and pathological counter
        healthy = len(df[df['healthy_status'] == 'healthy'])
        pathological = len(df[df['healthy_status'] == 'pathological'])

        if "English" not in languages:
            languages["English"] = {"healthy": healthy, "pathological": pathological}
        else:
            languages["English"]["healthy"] += healthy

        return languages
    
    def all_languages_counter(self) -> dict:
        languages = self.czech_language_counter()

        english_count = self.english_language_counter()
        for lang, counts in english_count.items():
            if lang in languages:
                languages[lang]['healthy'] += counts['healthy']
                languages[lang]['pathological'] += counts['pathological']
            else:
                languages[lang] = counts

        german_count = self.german_language_counter()
        for lang, counts in german_count.items():
            if lang in languages:
                languages[lang]['healthy'] += counts['healthy']
                languages[lang]['pathological'] += counts['pathological']
            else:
                languages[lang] = counts

        common_voice_count = self.common_voice_language_counter()
        for lang, counts in common_voice_count.items():
            if lang in languages:
                languages[lang]['healthy'] += counts['healthy']
            else:
                languages[lang] = counts

        return languages

    def plot_statistics(self, languages: dict) -> None: 
        labels = list(languages.keys())
        healthy = [lang["healthy"] for lang in languages.values()]
        pathological = [lang["pathological"] for lang in languages.values()]
        
        total_samples = [h + p for h, p in zip(healthy, pathological)]

        plt.figure(figsize=(10, 7))
        plt.pie(total_samples, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Total Samples Distribution by Language')
        plt.savefig('./png/total_samples.png')
        plt.close()

        # Function to filter out 0% slices
        def filter_data(data, labels):
            filtered_data = [(d, l) for d, l in zip(data, labels) if d > 0]
            return [d for d, l in filtered_data], [l for d, l in filtered_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        healthy_filtered, labels_filtered_healthy = filter_data(healthy, labels)
        ax1.pie(healthy_filtered, labels=labels_filtered_healthy, autopct='%1.1f%%', startangle=140)
        ax1.set_title('Healthy Samples Distribution')

        pathological_filtered, labels_filtered_pathological = filter_data(pathological, labels)
        ax2.pie(pathological_filtered, labels=labels_filtered_pathological, autopct='%1.1f%%', startangle=140)
        ax2.set_title('Pathological Samples Distribution')

        plt.savefig('./png/healthy_pathological_samples.png')
        plt.close()

    def dataframe_from_excel(self, excel_path: str) -> pd.DataFrame:
        return pd.read_excel(excel_path)
    
    def gender_statistic_png(self) -> plt.figure:
        dataframe: pd.DataFrame = self.dataframe_from_excel("combined_languages.xlsx")

        # read 'gender' column
        genders = dataframe['gender'].value_counts()

        # bar plot
        plt.figure(figsize=(10, 7))
        plt.bar(genders.index, genders.values, color=['blue', 'pink', 'green'])
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.title('Gender Distribution, all datasets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('./png/gender_statistics.png')
        plt.close()

    def audio_files_length_histogram(self, dataframe: pd.DataFrame) -> plt.figure:
        file_paths = dataframe['file_path']

        # get audio files length
        audio_files_length = []
        corrupted_files_count = 0

        for file_path in file_paths:
            try:
                audio, _ = librosa.load(file_path)
                audio_files_length.append(len(audio))
            except:
                corrupted_files_count += 1

        # histogram
        plt.figure(figsize=(10, 7))
        plt.hist(audio_files_length, bins=50, color='blue', alpha=0.7)
        plt.xlabel('Audio file length')
        plt.ylabel('Count')
        plt.title('Audio files length histogram')
        plt.tight_layout()
        
        plt.savefig('./png/audio_files_length_histogram.png')
        plt.close()