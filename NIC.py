'''
File to define the model structure of NIC, based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf

model: Define the NIC training model

greedy_inference_model: Define the model used in greedy search.
                        Please initialize it with trained NIC model by load_weights()

image_dense_lstm: Define the encoding part of model used in beam search
                  Please initialize it with trained NIC model by load_weights()

text_emb_lstm: Define the decoding part of model used in beam search
               Please initialize it with trained NIC model by load_weights()
'''

import numpy as np
from keras import backend as K, Sequential
from keras import regularizers
from keras.initializers import glorot_uniform
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Embedding,
                          Input, Lambda, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Activation, Add,
                          AveragePooling2D, ZeroPadding2D)
from keras.models import Model

unit_size = 4096
lstm_size = 2048
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# Identity block

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X
    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid", name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), name=conv_name_base + '2b', padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c', padding="valid",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '1', padding="valid",
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation("relu")(X)

    return X


def ResNet50(inputs):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(inputs)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[256, 256, 2048], stage=5, block='a', s=3)
    X = identity_block(X, 3, filters=[256, 256, 2048], stage=5, block='b')
    X = identity_block(X, 3, filters=[256, 256, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)

    return X


def model(vocab_size, max_len, reg):
    resizeDim = (256, 256)

    inputs1 = Input(shape=(resizeDim[1], resizeDim[0], 3))

    encoded_image = ResNet50(inputs1)
    # X_img = Dropout(0.1)(X_img)
    X_img = BatchNormalization(name='batch_normalization_img')(encoded_image)
    X_img = Dense(lstm_size, activation='relu', kernel_regularizer=regularizers.l2(reg))(X_img)

    X_img = Lambda(lambda x: K.expand_dims(x, axis=1))(X_img)
    Model(inputs1, X_img).summary()

    # Text embedding
    inputs2 = Input(shape=(max_len,))
    X_text = Embedding(vocab_size, lstm_size, mask_zero=True, name='emb_text')(inputs2)
    Model(inputs2, X_text).summary()

    # Initial States
    a0 = Input(shape=(lstm_size,))
    c0 = Input(shape=(lstm_size,))

    LSTMLayer = LSTM(lstm_size, return_sequences=True, return_state=True, dropout=0, name='lstm')

    # Take image embedding as the first input to LSTM
    _, a, c = LSTMLayer(X_img, initial_state=[a0, c0])

    A, _, _ = LSTMLayer(X_text, initial_state=[a, c])
    output = TimeDistributed(Dense(vocab_size, activation='softmax',
                                   kernel_regularizer=regularizers.l2(reg),
                                   bias_regularizer=regularizers.l2(reg)), name='time_distributed_softmax')(A)

    return Model(inputs=[inputs1, inputs2, a0, c0], outputs=output, name='NIC')


def greedy_inference_model(vocab_size, max_len):
    EncoderDense = Dense(unit_size, use_bias=False, name='dense_img')
    EmbeddingLayer = Embedding(vocab_size, unit_size, mask_zero=True, name='emb_text')
    LSTMLayer = LSTM(unit_size, return_state=True, name='lstm')
    SoftmaxLayer = Dense(vocab_size, activation='softmax', name='time_distributed_softmax')
    BatchNormLayer = BatchNormalization(name='batch_normalization_img')

    # Image embedding
    inputs1 = Input(shape=(94208,))
    X_img = EncoderDense(inputs1)
    X_img = BatchNormLayer(X_img)
    X_img = Lambda(lambda x: K.expand_dims(x, axis=1))(X_img)

    # Text embedding
    inputs2 = Input(shape=(1,))
    X_text = EmbeddingLayer(inputs2)

    # Initial States
    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, _, c = LSTMLayer(X_img, initial_state=[a0, c0])

    x = X_text

    outputs = []
    for i in range(max_len):
        a, _, c = LSTMLayer(x, initial_state=[a, c])
        output = SoftmaxLayer(a)
        outputs.append(output)
        x = Lambda(lambda x: K.expand_dims(K.argmax(x)))(output)
        x = EmbeddingLayer(x)

    return Model(inputs=[inputs1, inputs2, a0, c0], outputs=outputs, name='NIC_greedy_inference_v2')


def image_dense_lstm():
    EncoderDense = Dense(unit_size, use_bias=False, name='dense_img')
    BatchNormLayer = BatchNormalization(name='batch_normalization_img')
    LSTMLayer = LSTM(unit_size, return_state=True, name='lstm')

    inputs = Input(shape=(94208,))
    X_img = EncoderDense(inputs)
    X_img = BatchNormLayer(X_img)
    X_img = Lambda(lambda x: K.expand_dims(x, axis=1))(X_img)

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, _, c = LSTMLayer(X_img, initial_state=[a0, c0])

    return Model(inputs=[inputs, a0, c0], outputs=[a, c])


def text_emb_lstm(vocab_size):
    EmbeddingLayer = Embedding(vocab_size, unit_size, mask_zero=True, name='emb_text')
    LSTMLayer = LSTM(unit_size, return_state=True, name='lstm')
    SoftmaxLayer = Dense(vocab_size, activation='softmax', name='time_distributed_softmax')

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))
    cur_word = Input(shape=(1,))

    X_text = EmbeddingLayer(cur_word)
    a, _, c = LSTMLayer(X_text, initial_state=[a0, c0])
    output = SoftmaxLayer(a)

    return Model(inputs=[a0, cur_word, c0], outputs=[output, a, c])
