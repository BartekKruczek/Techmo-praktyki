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
        for file_path in utils_handler.audio_path_iterator(combined_df):
            try:
                y, sr = librosa.load(file_path, sr=None)
                print(y, sr)
            except Exception as e:
                print(f"Error: {e}")
                continue