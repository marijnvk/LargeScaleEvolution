import copy
import glob
import os
import pickle
import random
import time

import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split

# Evolutionary context for convolutional neural networks on the CIFAR-10 data set.
# See https://arxiv.org/abs/1703.01041 for details.
class CNNInterface(object):

    # Optionally provdide a the location of the CIFAR-10 batches. If not provided,
    # Keras will download the data set.
    def __init__(self, data_dir=None):
        self.data = self.__load_data(data_dir)

    # Initial individuals contain no layers in the backbone, and have a default learning rate.
    def generate_init(self):
        return {'layers': [], 'skips': [], 'lr': 0.1}

    def read_dna(self, individual):
        try:
            with open(f"{individual}/dna.pkl", 'rb') as dna_file:
                return pickle.load(dna_file)
        except:
            return None

    def store_dna(self, individual, dna):
        with open(f"{individual}/dna.pkl", 'wb') as dna_file:
            pickle.dump(dna, dna_file, pickle.HIGHEST_PROTOCOL)

    def read_metrics(self, individual):
        try:
            with open(f"{individual}/metrics.pkl", 'rb') as metrics_file:
                return pickle.load(metrics_file)
        except:
            return None

    def store_metrics(self, individual, metrics):
        # TODO store model and weights
        # TODO weight sharing
        del metrics["model"]

        with open(f"{individual}/metrics.pkl", 'wb') as metrics_file:
            pickle.dump(metrics, metrics_file, pickle.HIGHEST_PROTOCOL)

    def evaluate_ind(self, dna):
        start_time = time.time()
        model = self.__create_model(dna)
        fitness = self.__train_model(model)
        evaluation_time = time.time() - start_time

        metrics = {}
        metrics["fitness"] = fitness
        metrics["evaluation_time"] = evaluation_time
        # return the model as part of the metrics so it can be stored
        metrics["model"] = model

        return metrics

    def has_better_fitness(self, fitness_one, fitness_two):
        return fitness_one > fitness_two

    # Apply a random mutation and return the resulting DNA. The method
    # will keep trying mutations on the original DNA until one succeeds.
    def mutate_ind(self, dna):
        mutations = [
            self.alter_learning_rate,
            self.identity,
            self.insert_convolution,
            self.remove_convolution,
            self.alter_stride,
            self.alter_numer_of_channels,
            self.filter_size,
            self.insert_one_to_one,
            self.add_skip,
            self.remove_skip
        ]
        success = False

        while not success:
            mutated_dna, success = random.choice(mutations)(dna)

        return mutated_dna

######## Mutations ########

    # Set the learning rate to a random value between half and double the current value.
    def alter_learning_rate(self, dna):
        mutated_dna = copy.deepcopy(dna)
        mutated_dna['lr'] = random.uniform(
            mutated_dna['lr'] / 2.0, mutated_dna['lr'] * 2.0)
        return mutated_dna, True

    # Don't change anything. In practice, this means "keep training".
    def identity(self, dna):
        return copy.deepcopy(dna), True

    # Insert a convolutional layer at a random spot in the backbone. Randomly also inserts
    # a batch normalization and a ReLU layer.
    def insert_convolution(self, dna):
        mutated_dna = copy.deepcopy(dna)
        # Where to insert the new layer(s)
        insert_index = random.randint(0, len(mutated_dna['layers']))
        # Random stride length in both directions
        stride_length = random.choice([1, 2])
        # Whether to include activations in the insertion
        apply_activations = random.choice([True, False])

        # Find the number of input channels for the new convolutional layer
        num_channels = self.data['x_train'].shape[3 if K.image_data_format(
        ) == 'channels_last' else 1]
        for i in range(insert_index):
            if mutated_dna['layers'][i][0] == 'c':
                num_channels = mutated_dna['layers'][i][3 if K.image_data_format(
                ) == 'channels_last' else 1]

        # Insert convolutional layer with 3x3 filters
        mutated_dna['layers'].insert(insert_index, ['c', float(
            num_channels), 3.0, 3.0, float(stride_length), float(stride_length)])
        if apply_activations:
            mutated_dna['layers'].insert(insert_index + 1, ['a', 'bn'])
            mutated_dna['layers'].insert(insert_index + 2, ['a', 'relu'])
        return mutated_dna, True

    # Remove a random convolutional layer, if possible
    def remove_convolution(self, dna):
        conv_indices = self.__get_conv_indices(dna)
        if len(conv_indices) == 0:
            return None, False
        else:
            mutated_dna = copy.deepcopy(dna)
            random_conv = random.choice(conv_indices)
            del mutated_dna['layers'][random_conv]
            return mutated_dna, True

    # Change the stride of a random convolutional layer in both directions, if possible.
    # The new value is randomly between half and double the current value.
    def alter_stride(self, dna):
        conv_indices = self.__get_conv_indices(dna)
        if len(conv_indices) == 0:
            return None, False
        else:
            mutated_dna = copy.deepcopy(dna)
            # Pick random convolutional layer
            random_conv = random.choice(conv_indices)
            # Determine the current stride length
            current_stride_len = mutated_dna['layers'][random_conv][4]
            new_stride_len = random.uniform(
                current_stride_len / 2.0, current_stride_len * 2.0)
            # Update the stride lengths
            mutated_dna['layers'][random_conv][4] = new_stride_len
            mutated_dna['layers'][random_conv][5] = new_stride_len
            return mutated_dna, True

    # Change the number of channels of a random convolutional layer, if possible.
    # The new value is randomly between half and double the current value.
    def alter_numer_of_channels(self, dna):
        conv_indices = self.__get_conv_indices(dna)
        if len(conv_indices) == 0:
            return None, False
        else:
            mutated_dna = copy.deepcopy(dna)
            random_conv = random.choice(conv_indices)
            current_num_channels = mutated_dna['layers'][random_conv][1]
            mutated_dna['layers'][random_conv][1] = random.uniform(
                current_num_channels / 2.0, current_num_channels * 2.0)
            return mutated_dna, True

    # Change the filter size of a random convolutional layer in a random direction, if possible.
    # The new value is randomly between half and double the current value.
    def filter_size(self, dna):
        conv_indices = self.__get_conv_indices(dna)
        if len(conv_indices) == 0:
            return None, False
        else:
            mutated_dna = copy.deepcopy(dna)
            random_conv = random.choice(conv_indices)
            random_filter_dim = random.choice([2, 3])
            current_filter_size = mutated_dna['layers'][random_conv][random_filter_dim]
            mutated_dna['layers'][random_conv][random_filter_dim] = random.uniform(
                current_filter_size / 2.0, current_filter_size * 2.0)
            return mutated_dna, True

    # Insert an identity layer at a random spot in the backbone. Randomly also inserts
    # a batch normalization and a ReLU layer.
    # Note: This mutation works slightly different in the paper, because the DNA in the paper
    #       is represented as a graph, as opposed to a list of layers and a list of skip
    #       connections as in this implementation.
    def insert_one_to_one(self, dna):
        mutated_dna = copy.deepcopy(dna)
        insert_index = random.randint(0, len(mutated_dna['layers']))
        mutated_dna['layers'].insert(insert_index, ['a', 'bn'])
        mutated_dna['layers'].insert(insert_index + 1, ['a', 'relu'])
        return mutated_dna, True

    # Add a skip connection between two random spots in the backbone, if possible.
    # The mutation will not be successful if all possible skip connections are already present
    # TODO implement mutation. The mutation must not be able to insert a skip connection that
    #      starts or ends between a BN and a ReLU layer.
    # TODO update insert/remove convolution mutations and one_to_one mutation to accomodate skip connections
    # TODO update model building to accomodate skip connections
    def add_skip(self, dna):
        return None, False

    # Remove a random skip connection, if possible
    def remove_skip(self, dna):
        if len(dna['skips']) == 0:
            return None, False
        mutated_dna = copy.deepcopy(dna)
        del mutated_dna['skips'][random.randint(
            0, len(mutated_dna['skips']) - 1)]
        return mutated_dna, True

######## Context-specific helper methods ########

    # Load the CIFAR-10 data set. If a data directory is not provided, the
    # Keras built-in functionality for downloading the data set is used.
    # The provided directory should contain the CIFAR-10 batch files as
    # provided on https://www.cs.toronto.edu/~kriz/cifar.html
    def __load_data(self, data_dir=None):
        if data_dir is None:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            x_train = np.zeros((50000, 3, 32, 32), dtype='uint8')
            y_train = np.zeros((50000,), dtype='uint8')

            for i in range(1, 6):
                fpath = os.path.join(data_dir, 'data_batch_' + str(i))
                data, labels = cifar10.load_batch(fpath)
                x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
                y_train[(i - 1) * 10000: i * 10000] = labels

            fpath = os.path.join(data_dir, 'test_batch')
            x_test, y_test = cifar10.load_batch(fpath)

            y_train = np.reshape(y_train, (len(y_train), 1))
            y_test = np.reshape(y_test, (len(y_test), 1))

        # Ready the data for use in networks
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        # Data comes as channel first. Transpose the dimensions if necessary
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        # Split off validation data in the ratio described in the paper
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1)

        return {
            'x_train': x_train,
            'x_test': x_test,
            'x_val': x_val,
            'y_train': y_train,
            'y_test': y_test,
            'y_val': y_val
        }

    # Get the indices of convolutional layers in the backbone
    def __get_conv_indices(self, dna):
        indices = []
        for i in range(len(dna['layers'])):
            if dna['layers'][i][0] == 'c':
                indices.append(i)
        return indices

    # Build a network model from the provided DNA, including the optimizer
    def __create_model(self, dna):
        m_in = Input(shape=self.data["x_train"].shape[1:])
        m_out = m_in

        # Build the network backbone
        for layer in dna['layers']:
            if layer[0] == 'c':
                # Values are stored as float in DNA and rounded when
                # creating layers so small mutational changes can stack
                # up to have an influence.
                # Filter size values are rounded to the nearest odd number.
                # Stride values in DNA are the log-base-2 values.
                m_out = Conv2D(
                    round(layer[1]), 
                    (
                        int(layer[2]) if int(layer[2]) % 2 != 0 else int(layer[2]) + 1,
                        int(layer[3]) if int(layer[3]) % 2 != 0 else int(layer[3]) + 1
                    ),
                    strides=(2**round(layer[4]), 2**round(layer[5]))
                )(m_out)
            elif layer[0] == 'a':
                if layer[1] == 'bn':
                    m_out=BatchNormalization(m_out)
                elif layer[1] == 'relu':
                    m_out=Activation('relu')(m_out)
                else:
                    raise ValueError(f"Unkown activation type: {layer[1]}")
            else:
                raise ValueError(f"Unknown layer type: {layer[0]}")

        # Add standard layers for classification
        m_out=MaxPooling2D(pool_size=(2, 2))(m_in)
        m_out=Flatten()(m_out)
        m_out=Dense(512)(m_out)
        m_out=Activation('relu')(m_out)
        m_out=Dropout(0.5)(m_out)
        m_out=Dense(10)(m_out)
        m_out=Activation('softmax')(m_out)

        # Compile the model with optimizer as described in the paper, including the learning rate
        model=Model(inputs=m_in, outputs=m_out)
        model.compile(optimizer=SGD(lr=dna['lr'], decay=0.0001, momentum=0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # TODO consider automatic validation data
    # TODO store trained models
    # TODO return fitness
    def __train_model(self, model):
        history = model.fit(self.data['x_train'], self.data['y_train'], batch_size=50,
                            epochs=30, verbose=0, validation_data=(self.data['x_val'], self.data['y_val']))
