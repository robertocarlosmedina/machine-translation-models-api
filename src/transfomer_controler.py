from transformer.transformer import Transformer_Translator as Transformer


class Transformer_Controller:

    def __init__(self) -> None:
        self.transformer_models = {
            "en-cv": Transformer("en", "cv"),
            "cv-en": Transformer("cv", "en"),
        }
    
    def get_model_parameters(self):
        return self.transformer_models["en-cv"].count_hyperparameters()
    
    def translate(self, direction: str, sentence: str) -> str:
        return self.transformer_models[direction].get_translation(sentence)