from flask_restful import Api, Resource, reqparse
from flask import Flask, jsonify
from flask_cors import CORS

from src.models_manager import Models_Manager


app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)

request_put_args = reqparse.RequestParser()
request_put_args.add_argument("sentence", type=str, help="Sentence to be translated.")
request_put_args.add_argument("model", type=str, help="Model to use on the translation.")
request_put_args.add_argument("source", type=str, help="Source languague on the translation.")
request_put_args.add_argument("target", type=str, help="Target languague on the translation.")

translator_manager = Models_Manager()


class Translation(Resource):

    def get(self):
        """
            GET Method to get all the model and their parameter numbers.
        """

        models_and_parameters = translator_manager.get_models_parameters()

        return jsonify(models_and_parameters)

    def post(self):
        """
            POST method to request one translation.
        """
        args = request_put_args.parse_args()

        translation = translator_manager.translate(
                                args["model"], args["source"],
                                args["target"], args["sentence"])

        return jsonify(translation)


class Resfull_API:
    @staticmethod
    def start():
        api.add_resource(Translation, "/translate")
        app.run(debug=False)


if __name__ == "__main__":

    Resfull_API.start()
