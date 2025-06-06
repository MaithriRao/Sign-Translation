
import math
import pickle
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch import Tensor
import io
import time
import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from typing import List
from sacrebleu.metrics import BLEU
import numpy as np
from sklearn.model_selection import KFold
from . import datasets
from .model import Model
from .utils import Tokenization


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128


def train_and_evaluate(fold, ds, tokenization, augment):

    if tokenization == Tokenization.SOURCE_ONLY:
        tokenization_dir = "source_only"
    elif tokenization == Tokenization.SOURCE_TARGET:
        tokenization_dir = "source_target"
    else:
        raise ValueError("Invalid tokenization value")

    if not augment:
        augment_dir = "original_data"
    else:
        augment_dir = "aug_data"

    save_folder = os.path.join("data_split/k_fold", tokenization_dir, augment_dir, "onlyGloss")
    save_file_path = os.path.join(save_folder, "result")
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    model = Model(ds, augment)

    (original, modified, full) = ds
    (tokenizer_original, vocab_original, sentences_original) = original
    (sentences_modified,) = modified
    (source_text_original, target_gloss_original) = sentences_original

    # Combine original source_sentences and target_sentences into a single dataset
    dataset = list(zip(source_text_original, target_gloss_original))

    # Split the dataset into 3 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(kf.split(dataset))

    # Split the dataset into train and test sets based on the current fold
    train_indices = [idx for fold_idx, idx in enumerate(folds[fold][0]) if fold_idx != fold]
    test_indices = folds[fold][1]
    train_data = [dataset[idx] for idx in train_indices]
    test_data = [dataset[idx] for idx in test_indices]

    # Split the train data into source and target
    train_source = [src for src, _ in train_data]
    train_target = [tgt for _, tgt in train_data]

    # Augment the training data if augment=True
    if augment:
        train_source, train_target = augment_data(train_source, train_target, sentences_modified)

    train_data = model.data_process(train_source, train_target, tokenization)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=model.generate_batch)

    # test_data = data_process([src for src, _ in test_data], [tgt for _, tgt in test_data])    
    # test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    NUM_EPOCHS = 1000
    loss_graf = []

    transformer = model.create_transformer()
    transformer = transformer.to(device)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    train_log = open(save_file_path+ f"_fold_{fold}_train_log.txt", 'w')

    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()

        train_loss = model.train_epoch(transformer, train_iter, optimizer)
        if not augment:
            train_loss = train_loss.tolist()

        end_time = time.time()
        log = "Epoch: " + str(epoch)+", Train loss: "+ str(train_loss)+" Epoch duration "+ str(end_time - start_time)+"\n"
        train_log.write(log)
        if epoch>1 and train_loss < min(loss_graf):
            torch.save(transformer.state_dict(), save_file_path+f"_fold_{fold}_best_model.pt")
            log = "min so far is at epoch: "+ str(epoch)+"\n"
            train_log.write(log)
            best_epoch = epoch

        loss_graf.append(train_loss)

    log = "best epoch is: "+ str(best_epoch)
    train_log.write(log)
    train_log.close()


    torch.save(transformer.state_dict(), save_file_path+f"_fold_{fold}_last_model.pt")

    # Evaluation
    ground_truth = []
    hypothesis = []
    num_P_T = 0
    num_T_P = 0
    num_e = 0

    for de_text, gl_text in test_data:
        if tokenization == Tokenization.SOURCE_TARGET:
            source_entry = de_text[1]
            target_entry = gl_text[1]

            print(f"Source Sententence : {source_entry}")
            print(f"Target Sententence : {target_entry}")
            

            gl_pred = model.translate(transformer, source_entry, model.text_vocab, model.mms_vocab, model.text_tokenizer, tokenization)
            print(f"gloss prediction   : {gl_pred}")

            translated_sentence = ""
            for char in gl_pred:
                if char == "▁":
                    translated_sentence += " "
                elif char != " ":
                    translated_sentence += char

            translated_sentence = translated_sentence.strip()

            print(f"translated_sentence: {translated_sentence}")

            ground_truth.append(target_entry)
            hypothesis.append(translated_sentence)

            P = len(translated_sentence.split())
            T = len(target_entry.split())

        elif tokenization == Tokenization.SOURCE_ONLY:
            source_entry = de_text[1]
            target_entry = "".join(gl_text[1])

            print(f"Source Sententence: {source_entry}")
            print(f"Target Sententence: {target_entry}")

            gl_pred = model.translate(transformer, source_entry, model.text_vocab, model.mms_vocab, model.text_tokenizer, tokenization)
            print(f"Predicted gloss   : {gl_pred}")

            ground_truth.append(target_entry)
            hypothesis.append(gl_pred)

            P = len(gl_pred.split())
            T = len(target_entry.split())

        else:
            raise ValueError("Invalid tokenization value")
  
        if P > T:
            print("P:", P)
            num_P_T += 1
        elif T > P:
            print("T:", T)
            num_T_P += 1
        else:
            num_e += 1

    with open(save_file_path + f"_fold_{fold}_outputs.txt", "w") as f:
        f.write(f"P>T: {num_P_T}\n")
        f.write(f"T>P: {num_T_P}\n")
        f.write(f"equal: {num_e}\n")

        refs = [ground_truth]
        bleu = BLEU()
        result = bleu.corpus_score(hypothesis, refs)
        f.write(f"BLEU score for maingloss: {result}\n")

    return result.score


def augment_data(train_source, train_target, sentences):
    (source_sentences, target_sentences) = sentences
    augmented_train_source = train_source.copy()
    augmented_train_target = train_target.copy()

    augmented_train_source.extend(source_sentences)
    augmented_train_target.extend(target_sentences)

    return augmented_train_source, augmented_train_target


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python k_fold.py [--source-only|--source-target]")
        sys.exit(1)

    if sys.argv[1] == "--source-only":
        print("Using source only")
        tokenization = Tokenization.SOURCE_ONLY
    elif sys.argv[1] == "--source-target":
        print("Using source and target")
        tokenization = Tokenization.SOURCE_TARGET
    else:
        print("You have to specify either --source-only or --source-target as an argument.")
        sys.exit(1)

    original_scores = []
    augmented_scores = []

    ds = datasets.read(tokenization)

    for fold in range(10):
        print(f"Current fold {fold}:")
        print("Original data :")
        original_score = train_and_evaluate(fold, ds, tokenization, augment=False)
        original_scores.append(original_score)

        print("Augmented data:")
        augmented_score = train_and_evaluate(fold, ds, tokenization, augment=True)
        augmented_scores.append(augmented_score)

    avg_original_score = np.mean(original_scores)
    avg_augmented_score = np.mean(augmented_scores)

    if tokenization == Tokenization.SOURCE_ONLY:
        tokenization_str = "source-only"
    elif tokenization == Tokenization.SOURCE_TARGET:
        tokenization_str = "source-target"
    else:
        raise ValueError("Invalid tokenization value")

    print(f"{tokenization_str} BLEU score on original data for each fold: {original_scores}")
    print(f"{tokenization_str} Average BLEU score on original data: {avg_original_score}")

    print(f"{tokenization_str} BLEU score on augmented data for each fold: {augmented_scores}")
    print(f"{tokenization_str} Average BLEU score on augmented data: {avg_augmented_score}")
