import librosa
import numpy as np

from utils import UtilsHandler

class FeatureExtractionHandler:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Klasa do ekstrakcji cech z plików audio"
    
    def get_MFCC(self):
            utils_handler = UtilsHandler("Database")
            combined_df = utils_handler.combined_language_pd()
            y_values = []
            sr_values = []
            mfcc_values = []

            for file_path in utils_handler.audio_path_iterator(combined_df):
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_array= np.asarray(mfcc)
                    mfcc_flattened = mfcc_array.flatten().astype(np.float64)

                    # print(f"MFCC {mfcc_flattened}")
                    # manage the length of the MFCC array using pad_or_truncate
                    # max_length = 173
                    # if len(mfcc) >= max_length:
                    #     mfcc = mfcc[:max_length]
                    # else:
                    #     mfcc = np.pad(mfcc, ((0, 0), (0, max_length - len(mfcc))), mode='constant')

                    y_values.append(y)
                    sr_values.append(sr)
                    mfcc_values.append(mfcc_flattened)
                except Exception as e:
                    print(f"Error: {e}")
                    y_values.append(None)
                    sr_values.append(None)
                    mfcc_values.append(None)

            # Use excel_updater to add y, sr, and MFCC columns to the Excel file
            # utils_handler.excel_updater(combined_df, y_values, sr_values, mfcc_values)

            return y_values, sr_values, mfcc_values