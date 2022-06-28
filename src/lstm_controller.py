from lstm.lstm import Seq2Seq_Translator as LSTM


class LSTM_Controller:

    def __init__(self) -> None:
        self.lstm_models = {
            "en-cv": LSTM("en", "cv"),
            "cv-en": LSTM("cv", "en"),
        }
    
    def get_model_parameters(self):
        return self.lstm_models["en-cv"].count_hyperparameters()
    
    def translate(self, direction: str, sentence: str) -> str:
        return self.lstm_models[direction].translate_sentence(sentence)
