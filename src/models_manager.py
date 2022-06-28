from sqlalchemy import true
from src.transfomer_controler import Transformer_Controller
from src.gru_controler import GRU_Controller
from src.lstm_controller import LSTM_Controller


class Models_Manager:

    def __init__(self) -> None:

        self.models = {
            "gru": GRU_Controller(),
            "lstm": LSTM_Controller(),
            "transformer": Transformer_Controller()
        }

        self.src_trg_list = ["en", "cv"]
    
    def get_models_parameters(self) -> dict:

        model_descrition = [{"name": key, "parameters": value.get_model_parameters()}
                for key, value in self.models.items()]

        return {"data": {"translator_models": model_descrition}}
    
    def translate(self, model: str, source: str, target: str, sentence) -> str:
        """
            Method to translate a sentence according to the model 
            source and target languague.
        """
        valid_model = self.model_exits(model)
        valid_src_trg = self.valid_source_and_target(source, target)
        valid_sentence = self.valid_sentence(sentence)

        if valid_sentence[0] and valid_model[0] and valid_src_trg[0]:
            translation = self.models[model].translate(f"{source}-{target}", sentence)
            return self.response_message(translation, [])
        
        return self.response_message(
            None, [valid_model[1], valid_sentence[1], valid_src_trg[1]])

    def model_exits(self, model: str) -> list:
        """
            Method to validate the given model name.
        """
        if model not in self.models.keys():
            return [False, "Model doesn't exist"]

        return [True, ""]
    
    def valid_source_and_target(self, source: str, target: str) -> list:
        """
            Method to validate the given source and target extentions
            to perform the translation.
        """
        if source != target and source in self.src_trg_list and \
                target in self.src_trg_list:
            return [True, ""]
        
        return [False, "Bad configuration on the source and target languague"]
    
    def valid_sentence(self, sentence: str) -> list:
        """
            Method to validate the given sentence to translate.
        """
        if sentence and sentence != " ":
            return [True, ""]
        
        return [False, "The sentence is an empty sentence"]

    def response_message(self, translation: str, errors=[]) -> dict:
        """
            Method to prepare the dict response for the translation 
            request.
        """
        return {
            "data": [
                {"translation": translation}
            ],
            "error": [
                {"id": i, "error": error} for i, error in enumerate(errors) if error != ""
            ]
        }
