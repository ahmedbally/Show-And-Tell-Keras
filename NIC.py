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
from keras import backend as K
from keras import regularizers
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Embedding,
                          Input, Lambda, TimeDistributed, Conv2D, MaxPooling2D, Flatten)
from keras.models import Model

unit_size = 1024


def model(vocab_size, max_len, reg):
    resizeDim = (192, 384)

    inputs1 = Input(shape=(resizeDim[1], resizeDim[0], 3))

    encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(inputs1)
    encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2))(encoder)

    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    X_img = MaxPooling2D((2, 2))(encoder)
    # Image embedding
    # inputs1 = Input(shape=(94208,))
    X_img = Dropout(0.1)(encoder)
    X_img = Flatten()(X_img)
    X_img = Dense(unit_size, use_bias=False,
                  kernel_regularizer=regularizers.l2(reg),
                  name='dense_img')(X_img)
    X_img = BatchNormalization(name='batch_normalization_img')(X_img)
    X_img = Lambda(lambda x: K.expand_dims(x, axis=1))(X_img)
    Model(inputs1, X_img).summary()

    # Text embedding
    inputs2 = Input(shape=(max_len,))
    X_text = Embedding(vocab_size, unit_size, mask_zero=True, name='emb_text')(inputs2)
    X_text = Dropout(0.1)(X_text)
    Model(inputs2, X_text).summary()

    # Initial States
    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    LSTMLayer = LSTM(unit_size, return_sequences=True, return_state=True, dropout=0.1, name='lstm')

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
