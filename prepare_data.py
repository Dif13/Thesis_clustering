import logging
import csv

import numpy as np
import pandas as pd
import easygui
import sys
import os
from sklearn.model_selection import train_test_split


def get_logger(name):
    _log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"

    def get_file_handler():
        file_handler = logging.FileHandler("x.log")
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter(_log_format))
        return file_handler

    def get_stream_handler():
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(_log_format))
        return stream_handler

    def_logger = logging.getLogger(name)
    def_logger.setLevel(logging.INFO)
    def_logger.addHandler(get_file_handler())
    def_logger.addHandler(get_stream_handler())
    return def_logger


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def create_ml_df_key_words_oecds(key_words_oecds_prepared, dict_keywords_file, dict_oecds_file):
    df_keywords_oecds = pd.read_csv(key_words_oecds_prepared, index_col=0)
    df_dict_keywords = pd.read_csv(dict_keywords_file)
    df_dict_oecds = pd.read_csv(dict_oecds_file)

    revers_dict_keyword = dict(zip(df_dict_keywords['uniq_keywords'], df_dict_keywords['index']))
    revers_dict_oecds = dict(zip(df_dict_oecds['uniq_oecds'], df_dict_oecds['index']))
    len_revers_dict_keyword = len(revers_dict_keyword)
    len_revers_dict_oecds = len(revers_dict_oecds)

    """

    tmp_ml_df_X = np.zeros((len(df_keywords_oecds['keyword_list.0']), len_revers_dict_keyword), bool)
    tmp_ml_df_Y = np.zeros((len(df_keywords_oecds['keyword_list.0']), len_revers_dict_oecds), bool)
    print(tmp_ml_df_X.__sizeof__())
    print(tmp_ml_df_Y.__sizeof__())
    for i, (keywords, oecds) in enumerate(zip(df_keywords_oecds['keyword_list.0'], df_keywords_oecds['oecds.0'])):
        tmp_X = np.zeros(len_revers_dict_keyword, bool)
        tmp_Y = np.zeros(len_revers_dict_oecds, bool)
        
        for keyword in keywords.split(sep=chr(0x1f)):
            if len(keyword) > 1:
                tmp_X[revers_dict_keyword[keyword]] = True

        for oecd in oecds.split(sep=chr(0x1f)):
            tmp_Y[revers_dict_oecds[oecd]] = True
        tmp_ml_df_X[i] = tmp_X
        tmp_ml_df_Y[i] = tmp_Y
        #del tmp

        #if i % 1000 == 0:
        #    print(i)
        #    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
        #                             key=lambda x: -x[1])[:10]:
        #        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        #    break
    print(f"tmp_ml_df_X.shape: {tmp_ml_df_X.shape}")
    print(f"tmp_ml_df_Y.shape: {tmp_ml_df_Y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(tmp_ml_df_X, tmp_ml_df_Y, test_size=0.20, random_state=42)
    np.save(root_dir + 'X_train.npy', X_train)
    np.save(root_dir + 'X_test.npy', X_test)
    np.save(root_dir + 'y_train.npy', y_train)
    np.save(root_dir + 'y_test.npy', y_test)
    #return tmp_ml_df_X, tmp_ml_df_Y
    print("array created")
    #column_name = [x for x in range(len_revers_dict_keyword)] + ['oecds.0']
    #print("column name created")
    #ml_df = pd.DataFrame(tmp_ml_df, columns=column_name)
    #print("df created")
    #ml_df.index.name = 'index_from_key_words_oecds_prepared'
    #print(ml_df.__sizeof__())
    #ml_df.to_csv(ml_df_name)
    """
    keywords_index = []
    oecd_index = []
    maxlen = {x: 0 for x in range(45)}
    for i, (keywords, oecds) in enumerate(zip(df_keywords_oecds['keyword_list.0'], df_keywords_oecds['oecds.0'])):
        tmp = []
        for keyword in keywords.split(sep=chr(0x1f)):
            if len(keyword) > 1:
                tmp.append(revers_dict_keyword[keyword])
        #maxlen = max(maxlen, len(tmp))
        maxlen[len(tmp)] += 1
        keywords_index.append(tmp)
        tmp = []
        for oecd in oecds.split(sep=chr(0x1f)):
            tmp.append((revers_dict_oecds[oecd]))
        oecd_index.append(tmp)

    df_keywords_oecds['keywords_index'] = keywords_index
    df_keywords_oecds['oecd_index'] = oecd_index
    #print(maxlen)
    # show_bar(maxlen)
    df_keywords_oecds.to_csv(key_words_oecds_prepared)
    # return maxlen
    return df_keywords_oecds
    # """


def show_bar(maxlen):
    import matplotlib.pyplot as plt

    x = [x for x in range(len(maxlen))]
    y = [maxlen[x] for x in range(len(maxlen))]

    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Диаграмма количества ключевых слов')
    ax.bar(x, y)

    ax.set_facecolor('seashell')
    fig.set_facecolor('floralwhite')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure
    ax.set_xlabel('Количество ключевых слов')
    ax.set_ylabel('Частота использования ключевых слов')

    plt.show()

def prepare_csv(df_name, output_file):
    """
    If function used separator: '' (ascii-code: 0x1f)
    :param df_name: csv-file
    :param output_file: scv-file
    :return: None
    """

    df = pd.read_csv(df_name, index_col=0)
    print(df.head(10))
    head = ['index', 'keyword_list.0', 'keyword_list.0_split']
    to_csv = []
    split_keywords = []
    split_oecds = []
    for i, (keyword, oecds) in enumerate(zip(df['keyword_list.0'], df['oecds.0'])):
        if keyword is np.NaN:
            split_keywords.append(np.NaN)
            continue
        comma = keyword.count(', ')
        semicolon = keyword.count('; ')
        sep = ', '
        if comma >= semicolon:
            pass
            # sep = ', '
        elif semicolon > max(comma, 2):
            sep = '; '

        temp = ''
        for k in keyword.split(sep=sep):
            temp += k.strip() + chr(0x1f)
        if temp:
            temp = temp[:-1]

        split_keywords.append(temp)

        temp = ''
        for oecd in oecds.split(sep=', '):
            temp += oecd.strip() + chr(0x1f)
        if temp:
            temp = temp[:-1]

        split_oecds.append(temp)

    # print(split_keywords[0])
    df['keyword_list.0'] = split_keywords
    df['oecds.0'] = split_oecds
    df.to_csv(output_file)


def create_dict(df_name, dict_keywords_file, dict_oecds_file):
    """
    If function used separator: '' (ascii-code: 0x1f)
    :param df_name:
    :param jupdict_file:
    :return:
    """
    df_source = pd.read_csv(df_name, index_col=0)
    df_dict_keywords = pd.DataFrame()
    df_dict_oecds = pd.DataFrame()
    list_uniq_keywords = []
    list_uniq_oecds = []
    polygonal_characteristic = {}

    for i, (keywords, oecds) in enumerate(zip(df_source['keyword_list.0'], df_source['oecds.0'])):
        k_split = []
        o_split = []
        try:
            k_split = keywords.split(sep=chr(0x1f))
        except:
            pass
        try:
            o_split = oecds.split(sep=chr(0x1f))
        except:
            pass
        list_uniq_keywords += k_split
        list_uniq_oecds += o_split

    list_uniq_oecds = sorted(list(set(list_uniq_oecds)))#, key=len)
    list_uniq_keywords = sorted(list(set(list_uniq_keywords)))#, key=len)
    if list_uniq_keywords[0] == '':
        list_uniq_keywords = list_uniq_keywords[1:]

    polygonal_characteristic = {x: [0]*len(list_uniq_oecds) for x in list_uniq_keywords}
    for i, (keywords, oecds) in enumerate(zip(df_source['keyword_list.0'], df_source['oecds.0'])):
        k_split = []
        o_split = []
        try:
            k_split = keywords.split(sep=chr(0x1f))
            o_split = oecds.split(sep=chr(0x1f))

            for k in k_split:
                for o in o_split:
                    #print(o)
                    ind = list_uniq_oecds.index(o)
                    #print(list_uniq_oecds.index(o))
                    polygonal_characteristic[k][ind] += 1
        except:
            continue


    # for i , m in enumerate(polygonal_characteristic['ВЫБОРЫ']):
    #     if m != 0:
    #         print(f"{list_uniq_oecds[i]}: {m}")

    characteristic = []
    for keyword in list_uniq_keywords:
        characteristic.append(polygonal_characteristic[keyword])

    df_dict_keywords.index.name = 'index'
    df_dict_oecds.index.name = 'index'
    df_dict_keywords['uniq_keywords'] = list_uniq_keywords
    df_dict_keywords['polygonal_characteristic'] = characteristic
    # list_uniq_oecds += [np.NaN] * (len(list_uniq_keywords) - len(list_uniq_oecds))
    df_dict_oecds['uniq_oecds'] = list_uniq_oecds
    df_dict_keywords.to_csv(dict_keywords_file)
    df_dict_oecds.to_csv(dict_oecds_file)


def create_new_csv(df_name, output_file):
    df_source = pd.read_csv(df_name)  # , sep=',')
    df = pd.DataFrame()
    # df[['keyword_list.0', 'oecds.0']] = df_source[['keyword_list.0', 'oecds.0']]
    df.index.name = 'index'

    dict_to_replace = {
        '$': '&',
        '?': ',',
        '^': ',',
        '_': '-',
        '•': ', ',
        'Ё': 'Е',
        'КЛЮЧЕВЫЕ СЛОВА:': ' ',
        'КЛЮЧЕВЫЕ СЛОВОСОЧЕТАНИЯ:': ' '
    }
    list_keyword = []
    oecds = []
    for keyword, oecd in zip(df_source['keyword_list.0'], df_source['oecds.0']):
        if keyword is np.NaN or oecd is np.NaN:
            # list_keyword.append(np.NaN)
            continue
        else:

            keyword = keyword.strip().upper()
            for key, value in dict_to_replace.items():
                keyword = keyword.replace(key, value)
            if keyword[0] in ',.:;':
                keyword = keyword[1:]
            if keyword[-1] in ',.:;':
                keyword = keyword[:-1]
        list_keyword.append(keyword.strip())
        oecds.append(oecd.strip().upper())

    df['keyword_list.0'] = list_keyword
    df['oecds.0'] = oecds
    df = df.reset_index(drop=True)
    df.to_csv(output_file)


def create_test_csv():
    NAME_CSV_FILES = ['dissertation', 'instruments_dis', 'keyword_dis', 'skvots_dis']

    for name_csv_file in NAME_CSV_FILES:

        with open(name_csv_file + '.csv', 'r') as csvread:
            reader = csv.reader(csvread)

            with open(name_csv_file + '_test.csv', 'w', newline='') as csvwrite:
                writer = csv.writer(csvwrite)
                for j, row in enumerate(reader):
                    writer.writerow(row)
                    # print(row)
                    if j == 3000:
                        break
        logger.info(f"Created test csv files: {name_csv_file}")


def open_explorer():
    # input_file = easygui.fileopenbox(default=root_dir, filetypes=["*.csv"])
    input_file = root_dir + "dissertation.csv"
    # input_file = root_dir + "dissertation_test.csv"
    # input_file = root_dir + "key_words_oecds.csv"
    # input_file = root_dir + "key_words_oecds_prepared.csv"
    return input_file


def main():
    logger.info(f"Start {__name__}")
    df_name = open_explorer()
    output_file = root_dir + "key_words_oecds.csv"
    key_words_oecds_prepared = root_dir + "key_words_oecds_prepared.csv"
    dict_keywords_file = root_dir + "dict_keywords.csv"
    dict_oecds_file = root_dir + "dict_oecds.csv"
    ml_df_name = root_dir + "ml_df.csv"
    # create_test_csv()
    # create_new_csv(df_name, output_file)
    # prepare_csv(output_file, key_words_oecds_prepared)
    # create_dict(key_words_oecds_prepared, dict_keywords_file, dict_oecds_file)
    create_ml_df_key_words_oecds(key_words_oecds_prepared, dict_keywords_file, dict_oecds_file)


if __name__ == '__main__':
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    # print(root_dir)
    logger = get_logger(__name__)
    main()
