from termcolor import colored
from tqdm import tqdm
import math
import numpy as np

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.meteor_score import meteor_score

import os
from pyter import ter
import random
import spacy

from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.seq2seq import Seq2Seq
from transformer.gammar_checker import Grammar_checker
from transformer.utils import display_attention, load_checkpoint, save_checkpoint,\
    translate_sentence, epoch_time, train, evaluate, check_dataset, progress_bar

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from nltk.translate.bleu_score import sentence_bleu


SEED = 1234
BATCH_SIZE = 10
HID_DIM = 256
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 3e-4
N_EPOCHS = 250
CLIP = 1


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Transformer_Translator:

    spacy_models = {
        "en": spacy.load("en_core_web_sm"),
        "pt": spacy.load("pt_core_news_sm"),
        "cv": spacy.load("pt_core_news_sm"),
    }

    def __init__(self, source_language: str, target_languague: str) -> None:

        self.source_languague = source_language
        self.target_languague = target_languague

        self.model = None
        self.optimizer = None
        self.criterion = None

        check_dataset()

        self.grammar = Grammar_checker()
        self.special_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']
        self.writer = SummaryWriter()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.get_dataset_data()
        self.setting_up_train_configurations()

    def get_dataset_data(self) -> None:

        self.SRC = Field(tokenize=self.tokenize_src,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True,
                         batch_first=True)

        self.TRG = Field(tokenize=self.tokenize_trg,
                         init_token='<sos>',
                         eos_token='<eos>',
                         lower=True,
                         batch_first=True)

        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(f".{self.source_languague}", f".{self.target_languague}"), 
            fields=(self.SRC, self.TRG),
            test="test", path=".data/crioleSet"
        )

        self.SRC.build_vocab(self.train_data, min_freq=2)
        self.TRG.build_vocab(self.train_data, min_freq=2)

        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=BATCH_SIZE,
            device=self.device
        )

        self.INPUT_DIM = len(self.SRC.vocab)
        self.OUTPUT_DIM = len(self.TRG.vocab)

        print(colored("=> Data has been collected and processed", 'cyan'))

    def tokenize_src(self, text: str):
        """
            Tokenizes Cap-Verdian text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_models[self.source_languague].tokenizer(text)]

    def tokenize_trg(self, text: str):
        """
            Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_models[self.target_languague].tokenizer(text)]

    def setting_up_train_configurations(self) -> None:
        enc = Encoder(self.INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      self.device)

        dec = Decoder(self.OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device)

        source_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        target_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]

        self.model = Seq2Seq(enc, dec, source_PAD_IDX,
                             target_PAD_IDX, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE)

        self.criterion = nn.CrossEntropyLoss(ignore_index=target_PAD_IDX)

        try:
            load_checkpoint(
                torch.load(
                    f"checkpoints/transformer-{self.source_languague}-{self.target_languague}.pth.tar",
                    map_location='cpu'),
                self.model, self.optimizer
            )
        except Exception as e:
            print("\n", e, "\n")
            print(colored("No existent checkpoint to load.", "red", attrs=["bold"]))

    def count_model_parameters(self) -> None:
        total_parameters =  sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(colored(f'\n==> The model has {total_parameters:,} trainable parameters\n', 'blue'))

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def show_train_metrics(self, epoch: int, epoch_time: str, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:

        print(f' Epoch: {epoch+1:03}/{N_EPOCHS} | Time: {epoch_time}')
        print(
            f' Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f' Val. Loss: {valid_loss:.3f} | Val Acc: {valid_accuracy:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
    
    def save_train_metrics(self, epoch: int, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        """
            Save the training metrics to be ploted in the tensorboard.
        """
        # All stand alone metrics
        self.writer.add_scalar(
            f"Training Loss ({self.source_languague}-{self.target_languague})", 
            train_loss, global_step=epoch)
        self.writer.add_scalar(
            f"Training Accuracy ({self.source_languague}-{self.target_languague})", 
            train_accuracy, global_step=epoch)
        self.writer.add_scalar(
            f"Validation Loss ({self.source_languague}-{self.target_languague})", 
            valid_loss, global_step=epoch)
        self.writer.add_scalar(
            f"Validation Accuracy ({self.source_languague}-{self.target_languague})", 
            valid_accuracy, global_step=epoch)
        
        # Mixing Train Metrics
        self.writer.add_scalars(
            f"Training Loss & Accurary ({self.source_languague}-{self.target_languague})", 
            {"Train Loss": train_loss, "Train Accurary": train_accuracy},
            global_step=epoch
        )

        # Mixing Validation Metrics
        self.writer.add_scalars(
            f"Validation Loss & Accurary  ({self.source_languague}-{self.target_languague})", 
            {"Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
        # Mixing Train and Validation Metrics
        self.writer.add_scalars(
            f"Train Loss & Validation Loss ({self.source_languague}-{self.target_languague})", 
            {"Train Loss": train_loss, "Validation Loss": valid_loss},
            global_step=epoch
        )
        self.writer.add_scalars(
            f"Train Accurary & Validation Accuracy ({self.source_languague}-{self.target_languague})",
            {"Train Accurary": train_accuracy, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )

    def train_model(self) -> None:

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            epoch = epoch + 1
            progress_bar = tqdm(
                total=len(self.train_iterator)+len(self.valid_iterator), 
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200
            )

            start_time = time.time()

            train_loss, train_accuracy = train(
                self.model, self.train_iterator, self.optimizer, self.criterion, 
                CLIP, epoch, progress_bar)

            valid_loss, valid_accuracy = evaluate(
                self.model, self.valid_iterator, self.criterion, epoch, progress_bar)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(
                    checkpoint, 
                    f"checkpoints/transformer-{self.source_languague}-{self.target_languague}.pth.tar")
            self.show_train_metrics(
                epoch, f"{epoch_mins}m {epoch_secs}s", train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )
            self.save_train_metrics(
                epoch, train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )

    def evalute_model(self) -> None:
        test_loss = evaluate(self.model, self.test_iterator, self.criterion)

        print(
            f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |'
        )

    def generate_confusion_matrix(self, src: str) -> None:
        translation, attention = translate_sentence(
            self.spacy_models[self.source_languague], src, self.SRC, self.TRG, self.model, self.device
        )

        print(f'Source (cv): {src}')
        print(colored(f'Predicted (en): {translation}', 'blue', attrs=['bold']))

        display_attention(self.spacy_models[self.source_languague], src, translation, attention)

    def test_model(self) -> None:
        test_data = self.get_test_data()
        os.system("clear")
        print("\n                  CV Creole Translator Test ")
        print("-------------------------------------------------------------\n")
        for data_tuple in test_data:
            src, trg = " ".join(
                data_tuple[0]), " ".join(data_tuple[1])
            translation, _ = translate_sentence(
                self.spacy_models[self.source_languague], src, self.SRC, self.TRG, self.model, self.device
            )
            print(f'  Source (cv): {src}')
            print(colored(f'  Target (en): {trg}', attrs=['bold']))
            print(colored(f'  Predicted (en): {self.untokenize_sentence(translation)}\n', 'blue', attrs=['bold']))

    def console_model_test(self) -> None:
        os.system("clear")
        print("\n                     CV Creole Translator ")
        print("-------------------------------------------------------------\n")
        while True:
            source = str(input(f'  Source (cv): '))
            translation, _ = translate_sentence(
                self.spacy_models[self.source_languague], source, self.SRC, self.TRG, self.model, self.device)

            print(
                colored(f'  Predicted (en): {self.untokenize_sentence(translation)}\n', 'blue', attrs=['bold'])
            )

    def get_translation(self, sentence: str) -> str:
        translation, _ = translate_sentence(
            self.spacy_models[self.source_languague], sentence, self.SRC, self.TRG, self.model, self.device)

        return self.untokenize_sentence(translation)

    def untokenize_sentence(self, tokens: list) -> str:
        """
            Method to untokenize the pedicted translation.
            Returning it on as an str, with some grammar checks.
        """
        tokens = self.remove_special_notation(tokens)
        if self.source_languague == "cv":
            translated_sentence = TreebankWordDetokenizer().detokenize(tokens)
            return self.grammar.check_sentence(translated_sentence)
        
        return " ".join(tokens)

    def remove_special_notation(self, sentence: list):
        return [token for token in sentence if token not in self.special_tokens]

    def get_test_data(self) -> list:
        return [(test.src, test.trg) for test in self.test_data.examples]

    def calculate_blue_score(self):
        """
            BLEU (bilingual evaluation understudy) is an algorithm for evaluating 
            the quality of text which has been machine-translated from one natural 
            language to another.
        """
        blue_scores = []
        len_test_data = len(self.test_data)

        for i, example in enumerate(self.test_data):
            src = vars(example)["src"]
            trg = vars(example)["trg"]
            predictions = []

            for _ in range(3):
                prediction, _ = translate_sentence(
                    self.spacy_models[self.source_languague], src, self.SRC, self.TRG, self.model, self.device)
                predictions.append(prediction[:-1])

            score = sentence_bleu(predictions, trg)
            blue_scores.append(score if score <= 1 else 1)

            progress_bar(i+1, len_test_data, f"BLUE score: {round(score, 8)}", "phases")

        score =  sum(blue_scores) /len(blue_scores)
        print(colored(f"\n\n==> Bleu score: {score * 100:.2f}\n", 'blue'))

    def calculate_meteor_score(self):
        """
            METEOR (Metric for Evaluation of Translation with Explicit ORdering) is 
            a metric for the evaluation of machine translation output. The metric is 
            based on the harmonic mean of unigram precision and recall, with recall 
            weighted higher than precision.
        """
        all_meteor_scores = []
        len_test_data = len(self.test_data)

        for i, example in enumerate(self.test_data):
            src = vars(example)["src"]
            trg = " ".join(vars(example)["trg"])
            predictions = []

            for _ in range(4):
                prediction, _ = translate_sentence(
                    self.spacy_models[self.source_languague], src, self.SRC, self.TRG, self.model, self.device)
                
                prediction = self.remove_special_notation(prediction)
                predictions.append(" ".join(prediction))

            score = meteor_score(predictions, trg)
            all_meteor_scores.append(score)

            progress_bar(i+1, len_test_data, f"METEOR score: {round(score, 8)}", "phases")

        score = sum(all_meteor_scores)/len(all_meteor_scores)
        print(colored(f"\n\n==> Meteor score: {score * 100:.2f}\n", 'blue'))

    def calculate_ter(self):
        """
            TER. Translation Error Rate (TER) is a character-based automatic metric for 
            measuring the number of edit operations needed to transform the 
            machine-translated output into a human translated reference.
        """
        all_translation_ter = 0
        len_test_data = len(self.test_data)

        for i, example in enumerate(self.test_data):
            src = vars(example)["src"]
            trg = vars(example)["trg"]

            prediction, _ = translate_sentence(
                self.spacy_models[self.source_languague], src, self.SRC, self.TRG, self.model, self.device)
            
            prediction = self.remove_special_notation(prediction)

            score = ter(prediction, trg)
            all_translation_ter += score
            progress_bar(i+1, len_test_data, f"TER score: {round(score, 8)}", "phases")

        print(colored(f"\n\n==>TER score: {all_translation_ter/len(self.test_data) * 100:.2f}\n", 'blue'))

    def count_hyperparameters(self) -> None:
        total_parameters =  sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_parameters
