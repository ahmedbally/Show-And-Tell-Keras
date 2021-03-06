'''
File to define data generator for training
'''
import cv2
import numpy as np
from keras.utils import to_categorical

from preprocessing.image import load_features
from preprocessing.text import load_dataset_token
from NIC import lstm_size
import os


def batch_generator(batch_size, max_len, tokenizer, dict_dir, dataset_dir, token_dir):
    vocab_size = tokenizer.num_words or (len(tokenizer.word_index) + 1)

    # img_features = load_features(dict_dir, dataset_dir, 1)
    img_ids = []
    with open(dataset_dir, 'r') as f:
        for line in f.readlines():
            img_ids.append(os.path.splitext(line)[0])

    raw_sentences = load_dataset_token(dataset_dir, token_dir, True)

    N = len(img_ids)
    while True:
        for i in range(0, N, batch_size):
            images = img_ids[i:i + batch_size]
            images_bytes = []
            for image in images:
                img_path = './datasets/Flickr8k_Dataset/' + image + '.jpg'
                resizeDim = (256, 256)
                img = cv2.imread(img_path)
                img = cv2.resize(img, resizeDim, interpolation=cv2.INTER_AREA)
                img = img.astype('float32') / 255
                # x = img.reshape(img.shape + (1,))
                images_bytes.append(img)
            sequences = tokenizer.texts_to_sequences(raw_sentences[i:i + batch_size])
            X_text = []
            Y_text = []
            for seq in sequences:
                if len(seq) > max_len:

                    X_text.append(seq[:max_len])
                    Y_text.append(seq[1:max_len + 1])
                else:
                    X_text.append(seq[:len(seq) - 1] + [0] * (max_len - len(seq) + 1))
                    Y_text.append(seq[1:] + [0] * (max_len - len(seq) + 1))

            X_text_mat = np.array(X_text)
            Y_text_mat = to_categorical(Y_text, vocab_size)
            yield ([np.array(images_bytes), X_text_mat, np.zeros([X_text_mat.shape[0], lstm_size]),
                    np.zeros([X_text_mat.shape[0], lstm_size])],
                   Y_text_mat)
