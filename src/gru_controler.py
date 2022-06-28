from gru.gru import Seq2Seq_Translator as GRU


class GRU_Controller:

    def __init__(self) -> None:
        self.gru_models = {
            "en-cv": GRU("en", "cv"),
            "cv-en": GRU("cv", "en"),
        }
    
    def get_model_parameters(self):
        return self.gru_models["en-cv"].count_hyperparameters()
    
    def translate(self, direction: str, sentence: str) -> str:
        return self.gru_models[direction].translate_sentence(sentence)
