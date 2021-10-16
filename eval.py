import numpy as np
# Set tensorflow to use CPU
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import tensorflow_model_optimization as tfmot
import cov2_genome
import pandas as pd
import lzma

alphabet = "acgt-"


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def string_to_one_hot_numpy(string):
    # Use Numpy where for speed
    one_hot = np.zeros((len(string), len(alphabet)), dtype=np.float32)
    for i, c in enumerate(string):
        if c in alphabet:
            one_hot[i, alphabet.index(c)] = 1
    return one_hot


def lineage_to_numpy(lineage):
    one_hot = np.zeros(len(all_lineages), dtype=np.float32)
    one_hot[lineage_to_index[lineage]] = 1
    return one_hot


import json

assignments = pd.read_csv("../pango-designation/lineages.csv")

metadata = pd.read_csv("oct_metadata_cut.tsv",
                       sep="\t",
                       usecols=["strain", "gisaid_epi_isl"])

#metadata.to_csv("oct_metadata_cut.tsv", sep="\t", index=False)

epi_to_name = dict(zip(metadata["gisaid_epi_isl"], metadata["strain"]))

name_to_taxon = dict(zip(assignments['taxon'], assignments['lineage']))

all_lineages = list(set(assignments['lineage']))
aliases = json.load(
    open("../pango-designation/pango_designation/alias_key.json", "rt"))


def get_multi_hot_from_lineage(lineage):
    multi_hot = np.zeros(len(all_lineages), dtype=np.float32)

    while True:
        if lineage in lineage_to_index:
            multi_hot[lineage_to_index[lineage]] = 1
        if "." in lineage:
            subparts = lineage.split(".")
            lineage = ".".join(subparts[:-1])
            continue
        elif lineage in aliases and aliases[lineage] != "":
            lineage = aliases[lineage]
            continue
        else:
            assert lineage == "A" or lineage == "B"
            break
    return multi_hot


all_epis = list(epi_to_name.keys())

lineage_to_index = dict(zip(all_lineages, range(len(all_lineages))))

del metadata
import random


def random_sample_from_list(the_list, proportion):
    return random.sample(the_list, int(proportion * len(the_list)))


num_shards = 200
all_shards = range(num_shards)
train_shards = random_sample_from_list(all_shards, 0.8)
test_shards = list(set(all_shards) - set(train_shards))

import gzip


def yield_examples(shards):
    while True:
        for shard_num in shards:
            file = open(f"shards/seq_shard_{shard_num}.tsv")
            for line in file:
                epi, seq = line.strip().split("\t")
                strain = epi_to_name[epi]
                if strain in name_to_taxon:
                    lineage = name_to_taxon[strain]
                    lineage_numpy = lineage_to_numpy(lineage)
                    yield (string_to_one_hot_numpy(seq), lineage_numpy)


import cov2_genome


def get_typed_examples(type):
    if type == "train":
        return yield_examples(train_shards)
    elif type == "test":
        return yield_examples(test_shards)


def yield_batch_of_examples(type, batch_size):
    example_iterator = get_typed_examples(type)
    while True:
        batch = [next(example_iterator) for _ in range(batch_size)]
        yield (np.stack([x[0] for x in batch]), np.stack([x[1]
                                                          for x in batch]))


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras import backend as K

import time

batch_size = 5

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# Load the model and do some inference
with tfmot.sparsity.keras.prune_scope():
    model = tf.keras.models.load_model("checkpoints/checkpoint.h5",
                                       custom_objects={
                                           "f1_m": f1_m,
                                           "precision_m": precision_m,
                                           "recall_m": recall_m
                                       })

test_iterator = yield_batch_of_examples("test", batch_size)
num_steps = 50
print(f"Making predictions for {num_steps*batch_size} sequences")
print(f"Starting time is {time.time()}")
starting_time = time.time()
results = model.predict(test_iterator, steps=num_steps)
# iterate through the results:
for i in range(num_steps * batch_size):
    print(f"Step {i}")
    print(results[i].shape)
print(f"Ending time is {time.time()}")
print(f"Time taken is {time.time() - starting_time}")
