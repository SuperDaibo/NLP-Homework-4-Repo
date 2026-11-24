import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        texts = load_lines(os.path.join(data_folder, f"{split}.nl"))

        preprocessed = []

        for text in texts:
            cleaned_txt = " ".join(text.lower().split())
            preprocessed.append(cleaned_txt)

        sql_quey = None
        if split != "test":
            sql_quey = load_lines(os.path.join(data_folder, f"{split}.sql"))
            assert len(texts) == len(sql_quey)

        self.encoder_ids = tokenizer(preprocessed,padding=False, truncation=False)["input_ids"]
        self.sql_ids = None
        if sql_quey is not None:
            self.sql_ids = tokenizer(sql_quey, padding = False, truncation = False)["input_ids"]

        num_example = len(self.encoder_ids)
        print(split, "number of examples:", num_example)


        length = [len(index) for index in self.encoder_ids]
        mean_sent_length = sum(length) / len(length)
        print(split, "mean sentence length:", mean_sent_length)

        counts = Counter()
        for sentence in self.encoder_ids:
            for token in sentence:
                counts[token] += 1
        vocab_size_nlp =  len(counts)
        print(split, "Vocabulary Size (NLP):", vocab_size_nlp)

        if self.sql_ids is not None:
            length_sql = [len(index) for index in self.sql_ids]
            mean_sql_length = sum(length_sql) / len(length_sql)
            print(split, "mean SQL query length:", mean_sql_length)

            counts_sql = Counter()
            for sentence in self.sql_ids:
                for token in sentence:
                    counts_sql[token] += 1
            vocab_size_sql =  len(counts_sql)

            print(split, "Vocabulary Size (SQL):", vocab_size_sql)

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.sql_ids is not None:
            return self.encoder_ids[idx], self.sql_ids[idx]
        else:
            return self.encoder_ids[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_seq = []
    sql_seq = []
    for example in batch:
        encoder_seq.append(example[0])
        sql_seq.append(example[1])

    encoder_tensor = [torch.tensor(ids, dtype = torch.long) for ids in encoder_seq]
    encoder_ids = pad_sequence(encoder_tensor, batch_first = True, padding_value = PAD_IDX)

    english_mask= []

    for ids in encoder_ids:
        row_temp = []
        for token in ids:
            if token != PAD_IDX:
                row_temp.append(1)
            else:
                row_temp.append(0)
        english_mask.append(row_temp)

    encoder_mask = torch.tensor(english_mask, dtype = torch.long)

    #SQL tokens the decoder receives as context
    decoder_input_tensors = []
    decoder_target_tensors = []

    for query in sql_seq:
        temp = [PAD_IDX] + query[:-1]
        decoder_input_tensors.append(torch.tensor(temp, dtype = torch.long))
        decoder_target_tensors.append(torch.tensor(query, dtype = torch.long))

    decoder_inputs = pad_sequence(decoder_input_tensors, batch_first = True, padding_value = PAD_IDX)

    #	The next SQL tokens the decoder should predict
    decoder_targets = pad_sequence(decoder_target_tensors, batch_first = True, padding_value = PAD_IDX)

    #	A starting token for each example when generating SQL
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_seq = batch
    encoder_tensor = [torch.tensor(ids, dtype = torch.long) for ids in encoder_seq]
    encoder_ids = pad_sequence(encoder_tensor, batch_first = True, padding_value = PAD_IDX)

    english_mask= []

    for ids in encoder_ids:
        row_temp = []
        for token in ids:
            if token != PAD_IDX:
                row_temp.append(1)
            else:
                row_temp.append(0)
        english_mask.append(row_temp)

    encoder_mask = torch.tensor(english_mask, dtype = torch.long)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
