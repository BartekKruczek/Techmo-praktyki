import librosa

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
                    mfcc = librosa.feature.mfcc(y=y, sr=sr).flatten()
                    y_values.append(y)
                    sr_values.append(sr)
                    mfcc_values.append(mfcc)
                except Exception as e:
                    print(f"Error: {e}")
                    y_values.append(None)
                    sr_values.append(None)
                    mfcc_values.append(None)

            # Use excel_updater to add y, sr, and MFCC columns to the Excel file
            utils_handler.excel_updater(combined_df, y_values, sr_values, mfcc_values)

            return y_values, sr_values, mfcc_values