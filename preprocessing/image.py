'''
Module to preprocess filckr8k image data
'''
import cv2
import numpy as np
import os
from _pickle import dump, load

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model

from PIL import Image


def load_images_as_arrays(directory):
    img_array_dict = {}
    for img_file in os.listdir(directory):
        img_path = directory + '/' + img_file

        img = Image.open(img_path)
        x = np.array(img)

        img_array_dict[os.path.splitext(img_file)[0]] = x

    return img_array_dict


def extract_features(directory):
    # base_model = InceptionV3(weights='imagenet')
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    #model = load_model('./preprocessing/CNN_encoder_100epoch.h5')
    #top = Flatten()(model.output)
    #model = Model(inputs=model.input, outputs=top)
    #print(model.summary())
    img_id = []
    img_matrices = []
    i = 0
    for img_file in os.listdir(directory):
        print(i, ":", i > 1999 and i < 8000 or i > 8999)
        '''if (i > 1999 and i < 8000 or i > 8999):
            i += 1
            continue'''
        img_path = directory + '/' + img_file
        resizeDim = (256, 512)
        img = cv2.imread(img_path)
        img = cv2.resize(img, resizeDim, interpolation=cv2.INTER_AREA)
        img = img.astype('float16') / 255
        #x = img.reshape(img.shape + (1,))
        img_id.append(os.path.splitext(img_file)[0])
        img_matrices.append(img)
        i += 1

    img_matrices = np.array(img_matrices)

    #img_features = model.predict(img_matrices, verbose=1)

    return {'ids': img_id, 'features': img_matrices}


def extract_feature_from_image(file_dir):
    img = image.load_img(file_dir, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # base_model = InceptionV3(weights='imagenet')
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    model = load_model('CNN_encoder_100epoch.h5')

    return model.predict(x)


def load_features(dict_dir, dataset_dir, repeat_times=1):
    assert (repeat_times >= 1)

    img_ids = []
    with open(dataset_dir, 'r') as f:
        for line in f.readlines():
            img_ids.append(os.path.splitext(line)[0])

    features_dict = load(open(dict_dir, 'rb'))
    #features_dict = extract_features('./datasets/Flickr8k_Dataset')
    dataset_features = []
    for img_id in img_ids:
        fidx = features_dict['ids'].index(img_id)
        dataset_features.append(np.vstack([features_dict['features'][fidx, :]] * repeat_times))

    #dataset_features = np.vstack(dataset_features)

    return np.array(dataset_features)


if __name__ == "__main__":
    # pre-extract image features from Inception Net
    image_directory = './datasets/Flickr8k_Dataset'
    features_dict = extract_features(image_directory)

    dump(features_dict, open('./datasets/features_dict2.pkl', 'wb'),protocol=4)
