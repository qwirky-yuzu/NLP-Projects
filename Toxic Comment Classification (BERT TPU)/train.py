# Import libraries
import tensorflow as tf
import logging
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    Dropout,
    Input,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import *
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

import argparse

# Instantiate TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)


# Clean data
class DataPreprocess:
    def __init__(self, train_data, sampled_amt, token_length, bz):
        parser = argparse.ArgumentParser(description="Set parameters")
        # Retrieve user input
        parser.add_argument("-sampled", type=int, help="Set amount of sampled data. Must be an integer", default=20000)
        parser.add_argument("-train", type=str, help="Set training dataset filepath")
        parser.add_argument("-token_len", type=int, help="Set max token length")
        parser.add_argument("-bz", type=int, help="Set batch size")

        args = parser.parse_args()
        
        
        self.train_data = args.train
        self.sampled_amt = args.sampled
        self.token_length = args.token_len
        self.bz = bz

    def read_data(self):
        train_df = pd.read_csv(self.train_data).fillna('blank')

    def clean_text(self, text):
        text = text.lower()

        # Split up contractions
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')

        # Split up toxic words
        toxic_reference_dictionary = {"youfuck": "you fuck", 
                                    "fucksex": "fuck sex",
                                    "bitchbot": "bitch bot",
                                    "offfuck": "fuck off",
                                    "donkeysex": "donkey sex",
                                    "securityfuck": "security fuck",
                                    "ancestryfuck": "ancestry fuck",
                                    "turkeyfuck": "turkey fuck",
                                    "faggotgay": "faggot gay",
                                    "fuckbot": "fuck bot",
                                    "assfuckers": "ass fucker",
                                    "ckckck": "cock",
                                    "fuckfuck": "fuck",
                                    "lolol": "lol",
                                    "pussyfuck": "pussy fuck",
                                    "gaygay": "gay",
                                    "haha": "ha",
                                    "sucksuck": "suck"
                                    }
        for existing,new_word in toxic_reference_dictionary.items():
            text = text.replace(existing,new_word)
        return text

        def cleaned_data(self):
            self.train_df['comment_text'] = self.train_df['comment_text'].map(lambda x : self.clean_text(x))

            return self.train_df

        def data_balancing(self):

            cleaned_training_dataset = self.cleaned_data()

            # Split dataset into toxic and clean comments
            LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            train_toxic = cleaned_training_dataset[cleaned_training_dataset[LABEL_COLUMNS].sum(axis=1) > 0]
            train_clean = cleaned_training_dataset[cleaned_training_dataset[LABEL_COLUMNS].sum(axis=1) == 0]

            # Sampling toxic and clean comments dataset
            final_training_df = pd.concat([
                train_toxic,
                train_clean.sample(self.sampled_amt)
            ])

            # Shuffle dataset
            final_training_df = final_training_df.sample(frac=1).reset_index(drop=True)

            return final_training_df, LABEL_COLUMNS

        def encode_data(self):
            
            final_training_df, LABEL_COLUMNS = self.data_balancing()

            # Name of the BERT model to use
            model_name = 'bert-base-uncased'

            # Max length of tokens
            max_length = self.token_length 

            # Load transformers config and set output_hidden_states to False
            config = BertConfig.from_pretrained(model_name)

            # Load BERT tokenizer
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

            def bert_encode(data):
                tokens = tokenizer.batch_encode_plus(
                    data,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding=True, 
                    return_tensors='tf',
                    return_token_type_ids = False,
                    return_attention_mask = True,
                    verbose = True
                )
                return tf.constant(tokens["input_ids"])

            # Split into training and validation dataset
            training_dataset, val_dataset = train_test_split(final_training_df, test_size=0.1, random_state=42)

            # Encode training and validation dataset - FEATURES
            train_encoded = bert_encode(training_dataset.comment_text)
            val_encoded = bert_encode(val_dataset.comment_text)

            # Encode training and validation dataset - LABELS
            train_labels=training_dataset[LABEL_COLUMNS].values
            train_labels=train_labels.reshape(-1,len(LABEL_COLUMNS))

            val_labels=val_dataset[LABEL_COLUMNS].values
            val_labels=val_labels.reshape(-1,len(LABEL_COLUMNS))

            print()
            print(f'Train labels shape: {train_labels.shape}')

            # Fitting into a Tensor unit
            train_dataset = (
                tf.data.Dataset.from_tensor_slices((train_encoded, train_labels))
                .shuffle(100)
                .batch(self.bz)
            ).cache()

            val_dataset = (
                tf.data.Dataset.from_tensor_slices((val_encoded, val_labels))
                .shuffle(100)
                .batch(self.bz)
            ).cache()

            return train_dataset, val_dataset


if __name__ == "main":

    # Fetch data
    train_dataset, val_dataset= DataPreprocess(train_data, sampled_amt, token_length, bz).encode_data()

    def bert_tpu_model(self):
    
        bert_encoder = TFAutoModel.from_pretrained('bert-base-uncased')

        input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')

        last_hidden_states = bert_encoder(input_ids)[0]

        clf_output = Flatten()(last_hidden_states)

        dense_01 = Dense(512, activation='relu', name='dense_01')(clf_output)
        dropout_01 = Dropout(0.5)(dense_01)

        dense_02 = Dense(512, activation='relu', name='dense_02')(dropout_01)
        dropout_02 = Dropout(0.5)(dense_02)

        out =Dense(6, activation='sigmoid', name='outputs')(dropout_02)

        model = Model(inputs=input_ids, outputs=out)

        return model

    with strategy.scope():
        model = bert_tpu_model()
        optimizer = Adam(learning_rate=1e-5, decay=1e-6)
        model.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        model.summary()

    history = History()

    # Set early stopping
    es = EarlyStopping(monitor='val_accuracy', 
                    mode='max', 
                    verbose=1, 
                    patience=8)  # Training will wait 8 epochs to check for any improvement to validation accuracy

    # Save best model to train epoch
    checkpoint = ModelCheckpoint('./bert_sampling_wgts.hdf5', 
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=True, 
                                mode='max');  

    callbacks = [es, checkpoint, history]

    model.fit(
        train_dataset,
        batch_size=self.bz,
        epochs=100,
        validation_data=val_dataset,
        verbose=1,
        callbacks=callbacks
    )


