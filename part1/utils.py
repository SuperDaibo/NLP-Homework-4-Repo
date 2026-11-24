import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    example_list = example["text"].split()
    neighbor = {"q": "wsa", "w": "qasde", "e": "wsdfr", "r": "edfgt", "t": "rfghy", "y": "tghju", "u": "yhjki", "i": "ujklo", "o": "iklp", "p": "ol", "a": "qwsz", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn", "j": "yuihknm", "k": "uiojlm", "l": "iopk", "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"}
    for i, word in enumerate(example_list):
      chance = random.random()
      if chance >= 0.7:
      # adding a typo
        random_idx = random.randrange(len(word))
        rand_letter = word[random_idx]

        dict_key = rand_letter.lower()
        if dict_key in neighbor: #random typo from neighbor
          typo = random.choice(neighbor[dict_key])

          if rand_letter.isupper(): #if upper
            typo = typo.upper()
          example_list[i]= (word[:random_idx]+typo+word[random_idx+1:])
    # put all word back into the original sentence
    example["text"] = " ".join(example_list)
    ##### YOUR CODE ENDS HERE ######

    return example
