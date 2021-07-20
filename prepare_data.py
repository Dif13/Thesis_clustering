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


def create_dict_csv(df_name):
    df_source = pd.read_csv(df_name, sep=',')
    df = pd.DataFrame()
    df[['keyword_list.0', 'oecds.0']] = df_source[['keyword_list.0', 'oecds.0']]

    for i in range(len(df.index)):
        if df.loc[i, 'keyword_list.0'] is np.NaN:
            df.loc[i, 'keyword_list.0'] = ''
        elif df.loc[i, 'oecds.0'] is np.NaN:
            df.loc[i, 'oecds.0'] = ''

    #df['keyword_list.0'] = df['keyword_list.0'].map(str.strip)
    #df['oecds.0'] = df['oecds.0'].map(str.strip)
    #sorted_key_words = ''
    sorted_key_words = ', '.join(df['keyword_list.0'].tolist()).upper().replace('.,', ',').replace(';', ',').split(', ')
    #for word in df['keyword_list.0'].tolist():
    #    sorted_key_words += word.strip() + ', '
    #sorted_key_words = sorted_key_words[:-2].upper().replace('.,', ',').replace(';', ',').split(', ')
    sorted_key_words = list(set(sorted_key_words))
    sorted_key_words.sort()
    if sorted_key_words[0] == '':
        del sorted_key_words[0]
    for i, word in enumerate(sorted_key_words):
        sorted_key_words[i] = word.strip()
    print(len(sorted_key_words))
    #sorted_clusters = ''
    sorted_clusters = ', '.join(df['oecds.0'].tolist()).upper().replace('.,', ',').replace(';', ',').split(', ')
    #for word in df['oecds.0']:
    #    sorted_clusters += word.strip() + ', '
    #sorted_clusters = sorted_clusters[:-2].upper().replace('.,', ',').replace(';', ',').split(', ')
    sorted_clusters = list(set(sorted_clusters))
    sorted_clusters.sort()
    if sorted_clusters[0] == '':
        del sorted_clusters[0]
    for i, word in enumerate(sorted_clusters):
        sorted_clusters[i] = word.strip()
    print(len(sorted_clusters))
    #df['keyword_list.0'] = df['keyword_list.0']
    #df['oecds.0'] = df['oecds.0'].map(str.strip)

    df_sorted_key_words = pd.DataFrame({'keyword_list.0': sorted_key_words})
    df_sorted_clusters = pd.DataFrame({'oecds.0': sorted_clusters})

    df_sorted_key_words.to_csv(root_dir + "sorted_key_words.csv")
    df_sorted_clusters.to_csv(root_dir + "sorted_clusters.csv")
    # TODO: last char is not [A-z][А-я][0-9]"')+-»I  like ,.
    # TODO: start char is ":"
    # TODO: Error in parse   ЗВЕЗД, ЛИГА, «ВПЕРЕД, ИТАЛИЯ»
    # TODO: is in alpha, beta, gamma, delta...
    # Ё is Е, sometimes ',' but not ', ' #41142 or sep='. ', or "   "
    # bad sort. Fix smth, but not all.
    return df



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
    input_file = root_dir + "dissertation.csv"
    #input_file = root_dir + "dissertation_test.csv"
    return input_file


def main():
    logger.info(f"Start {__name__}")
    df_name = open_explorer()
    #create_test_csv()
    df = create_dict_csv(df_name)


if __name__ == '__main__':
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    #print(root_dir)
    logger = get_logger(__name__)
    main()
