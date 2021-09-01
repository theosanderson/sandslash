import numpy as np
from tensorflow.python.ops.gen_linalg_ops import batch_cholesky
import pickle
if True:
    f = open("out.txt", "rt")

    def process_diff(x):
        position, residue = x.split(":")
        position = int(position)
        return (position, residue)

    info = {}
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print(i)
        epi, diffs = line.split("\t")
        if len(diffs) > 2:
            diffs = [process_diff(x) for x in diffs.split(",")]
        else:
            diffs = []
        info[epi] = diffs

    import cov2_genome
    import pandas as pd
    print('3')

    assignments = pd.read_csv(
        "https://raw.githubusercontent.com/cov-lineages/pango-designation/master/lineages.csv"
    )
    print('3b')
    metadata = pd.read_csv("../metadata.tsv", sep="\t")

    epi_to_name = dict(zip(metadata["Accession ID"], metadata["Virus name"]))

    name_to_taxon = dict(zip(assignments['taxon'], assignments['lineage']))

    all_lineages = list(set(assignments['lineage']))
    print('4')

    #pickle.dump([epi_to_name, name_to_taxon, info, all_lineages],
    #   open("data.pkl", "wb"))

#epi_to_name, name_to_taxon, info, all_lineages = pickle.load(
#   open("data.pkl", "rb"))

all_epis = list(info.keys())

lineage_to_index = dict(zip(all_lineages, range(len(all_lineages))))

del metadata
import random


def random_sample_from_list(the_list, proportion):
    return random.sample(the_list, int(proportion * len(the_list)))


train_epis = set(random_sample_from_list(all_epis, 0.8))
test_epis = set(all_epis) - train_epis


def yield_examples(epis):
    while True:
        random_ordered_epis = [x for x in random.sample(epis, len(epis))]
        for epi in random_ordered_epis:
            yield (diffs_to_numpy(info[epi]),
                   lineage_to_numpy(name_to_taxon[epi_to_name[epi].replace(
                       "hCoV-19/", "")]))


def lineage_to_numpy(lineage):
    one_hot = np.zeros(len(all_lineages), dtype=np.float32)
    one_hot[lineage_to_index[lineage]] = 1
    return one_hot


alphabet = ['a', 'c', 'g', 't', '-']

import cov2_genome


def sequence_to_numpy_sequence(seq):
    one_hot = np.zeros((len(seq), len(alphabet)), dtype=np.float32)
    for i, c in enumerate(seq):
        if c in alphabet:
            one_hot[i, alphabet.index(c)] = 1
    return one_hot


reference = sequence_to_numpy_sequence(cov2_genome.seq)


def yield_batch_of_examples(epis, batch_size):
    example_iterator = yield_examples(epis)
    while True:
        batch = [next(example_iterator) for _ in range(batch_size)]
        yield (np.stack([x[0] for x in batch]), np.stack([x[1]
                                                          for x in batch]))


def diffs_to_numpy(diffs):
    answer = reference.copy()
    for position, residue in diffs:
        answer[position, :] = 0
        if residue in alphabet:
            answer[position, alphabet.index(residue)] = 1
    return answer


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
input = tf.keras.Input(shape=(len(cov2_genome.seq), len(alphabet)),
                       dtype=tf.float32)
x = Flatten()(input)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(all_lineages), activation='softmax')(x)

model = tf.keras.Model(inputs=input, outputs=x)

print(model.summary())
opt = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 32

model.fit_generator(yield_batch_of_examples(train_epis, batch_size),
                    steps_per_epoch=2000,
                    epochs=10,
                    validation_data=yield_batch_of_examples(
                        test_epis, batch_size),
                    validation_steps=50)
