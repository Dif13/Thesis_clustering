import keras

import prepare_data
import os
import sys
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense, Flatten
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

MAXLEN = 10
MAX = 44


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def prepare_characteristic(df_dict_keywords):
    keys = df_dict_keywords.keys()
    print(keys[0])
    values = df_dict_keywords.values
    for i, val in enumerate(values):
        values[i] = list(map(int, val[1:-1].split(sep=', ')))
    dict_characteristic = dict(zip(keys, values))
    return dict_characteristic


def normalization(l):
    s = sum(l)
    if s == 0:
        return l
    else:
        return [x / s for x in l]


def prepare_ml_data():
    df_keywords_oecds = pd.read_csv(key_words_oecds_prepared, index_col=0)
    df_dict_oecds = pd.read_csv(dict_oecds_file)
    df_dict_keywords = pd.read_csv(dict_keywords_file)
    revers_dict_keyword = dict(zip(df_dict_keywords['uniq_keywords'], df_dict_keywords['index']))
    revers_dict_oecds = dict(zip(df_dict_oecds['uniq_oecds'], df_dict_oecds['index']))
    dict_characterictic = prepare_characteristic(df_dict_keywords['polygonal_characteristic'])
    print(dict_characterictic[0])
    # t_list = list(map(int, str(df_dict_keywords['polygonal_characteristic'])[1: -1].split(sep=', ')))
    # dict_keywords_characteristic = dict(zip(df_dict_keywords['uniq_keywords'], t_list))
    len_revers_dict_oecds = len(revers_dict_oecds)
    y = np.array([np.zeros(len_revers_dict_oecds, bool)] * len(df_keywords_oecds['oecds.0']))
    X = np.array([[np.zeros(len_revers_dict_oecds, float)] * MAXLEN] * len(df_keywords_oecds))

    for i, (keywords, oecds) in enumerate(zip(df_keywords_oecds['keyword_list.0'], df_keywords_oecds['oecds.0'])):
        for j, oecd in enumerate(oecds.split(sep=chr(0x1f))):
            y[i][revers_dict_oecds[oecd]] = True
            # print(revers_dict_oecds[oecd])

        tmp_X = []
        # print(tmp_X.__sizeof__() // 1024 // 1024)
        for k, keyword in enumerate(keywords.split(sep=chr(0x1f))):
            frequency = 0
            # revers_dict_keyword[keyword]
            # dict_characterictic[revers_dict_keyword[keyword]]
            try:
                tmp_X.append(dict_characterictic[revers_dict_keyword[keyword]])
            except KeyError as e:
                print(e)
            tmp_X.sort(key=sum)
        if len(tmp_X) < MAXLEN:
            tmp_X += [[0] * len_revers_dict_oecds] * (MAXLEN - len(tmp_X))
        elif len(tmp_X) > MAXLEN:
            tmp_X = tmp_X[:10]

        # print(tmp_X)
        tmp_X = np.array(list(map(normalization, tmp_X)))

        if len(tmp_X) != MAXLEN:
            breakpoint()
        np.random.shuffle(tmp_X)
        X[i] = tmp_X

    #     #Y[i] = tmp_Y
    # #print(len(df_dict_keywords))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    with open(root_dir + 'X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)
    with open(root_dir + 'X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)
    with open(root_dir + 'y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)
    with open(root_dir + 'y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)


def analyze_data(df_keywords_oecds):
    lst = list(df_keywords_oecds['keywords_index'])


if __name__ == '__main__':
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    key_words_oecds_prepared = root_dir + "key_words_oecds_prepared.csv"
    dict_keywords_file = root_dir + "dict_keywords.csv"
    dict_oecds_file = root_dir + "dict_oecds.csv"

    # df_keywords_oecds = prepare_data.create_ml_df_key_words_oecds(key_words_oecds_prepared, dict_keywords_file, dict_oecds_file)
    # analyze_data(df_keywords_oecds)
    prepare_ml_data()

