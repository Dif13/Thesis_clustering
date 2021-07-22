import logging
import csv

import numpy as np
import pandas as pd
import easygui
import sys
import os


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
    split = []
    for i, keyword in enumerate(df['keyword_list.0']):
        if keyword is np.NaN:
            split.append(np.NaN)
            continue
        comma = keyword.count(', ')
        semicolon = keyword.count('; ')
        sep = ', '
        if comma >= semicolon:
            pass
            #sep = ', '
        elif semicolon > max(comma, 2):
            sep = '; '

        temp = ''
        for k in keyword.split(sep=sep):
            temp += k.strip() + chr(0x1f)
        if temp:
            temp = temp[:-1]
        split.append(temp)
    print(split[0])
    df['keyword_list.0'] = split

    df.to_csv(output_file)


def create_dict(df_name, dict_file):
    """
    If function used separator: '' (ascii-code: 0x1f)
    :param df_name:
    :param dict_file:
    :return:
    """
    df_source = pd.read_csv(df_name, index_col=0)
    df_dict = pd.DataFrame()
    list_uniq_keywords = []
    for i, keyword in enumerate(df_source['keyword_list.0']):
        list_uniq_keywords += keyword.split(sep=chr(0x1f))
        if i % 1000 == 0:
            print(i)

    list_uniq_keywords = sorted(list(set(list_uniq_keywords)))
    df_dict.index.name = 'index'
    df_dict['uniq_keywords'] = list_uniq_keywords
    list_uniq_oecds = sorted(list(set(df_source['oecds.0'])))
    list_uniq_oecds += [np.NaN] * (len(list_uniq_keywords) - len(list_uniq_oecds))
    df_dict['uniq_oecds'] = list_uniq_oecds
    df_dict.to_csv(dict_file)







def create_new_csv(df_name, output_file):
    df_source = pd.read_csv(df_name)# , sep=',')
    df = pd.DataFrame()
    #df[['keyword_list.0', 'oecds.0']] = df_source[['keyword_list.0', 'oecds.0']]
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
            #list_keyword.append(np.NaN)
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
                    if j == 10:
                        break
        logger.info(f"Created test csv files: {name_csv_file}")


def open_explorer():
    # input_file = easygui.fileopenbox(default=root_dir, filetypes=["*.csv"])
    # input_file = root_dir + "dissertation.csv"
    # input_file = root_dir + "dissertation_test.csv"
    #input_file = root_dir + "key_words_oecds.csv"
    input_file = root_dir + "key_words_oecds_prepared.csv"
    return input_file


def main():
    logger.info(f"Start {__name__}")
    df_name = open_explorer()
    output_file = root_dir + "key_words_oecds.csv"
    output_file_split = root_dir + "key_words_oecds_prepared.csv"
    dict_file = root_dir + "dict_keywords_oecds.csv"
    # create_test_csv()
    # create_new_csv(df_name, output_file)
    # prepare_csv(output_file, output_file_split)
    create_dict(df_name, dict_file)


if __name__ == '__main__':
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    # print(root_dir)
    logger = get_logger(__name__)
    main()
