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


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def create_ml_df_key_words_oecds(key_words_oecds_prepared, dict_keywords_file, dict_oecds_file, ml_df_name):
    df_keywords_oecds = pd.read_csv(key_words_oecds_prepared, index_col=0)
    df_dict_keywords = pd.read_csv(dict_keywords_file)
    df_dict_oecds = pd.read_csv(dict_oecds_file)

    revers_dict_keyword = dict(zip(df_dict_keywords['uniq_keywords'], df_dict_keywords['index']))
    revers_dict_oecds = dict(zip(df_dict_oecds['uniq_oecds'], df_dict_oecds['index']))
    len_revers_dict_keyword = len(revers_dict_keyword)

    """

    tmp_ml_df = np.zeros((len(df_keywords_oecds['keyword_list.0']), len_revers_dict_keyword + 1), bool)
    print(tmp_ml_df.__sizeof__())
    for i, (keywords, oecd) in enumerate(zip(df_keywords_oecds['keyword_list.0'], df_keywords_oecds['oecds.0'])):
        tmp = np.zeros(len_revers_dict_keyword + 1, bool)
        for keyword in keywords.split(sep=chr(0x1f)):
            if len(keyword) > 1:
                tmp[revers_dict_keyword[keyword]] = True
        tmp[-1] = revers_dict_oecds[oecd]
        tmp_ml_df[i] = tmp
        #del tmp

        #if i % 1000 == 0:
        #    print(i)
        #    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
        #                             key=lambda x: -x[1])[:10]:
        #        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        #    break
    print("array created")
    column_name = [x for x in range(len_revers_dict_keyword)] + ['oecds.0']
    print("column name created")
    ml_df = pd.DataFrame(tmp_ml_df, columns=column_name)
    print("df created")
    ml_df.index.name = 'index_from_key_words_oecds_prepared'
    print(ml_df.__sizeof__())
    #ml_df.to_csv(ml_df_name)
    """
    keywords_index = []
    oecd_index = []
    for i, (keywords, oecd) in enumerate(zip(df_keywords_oecds['keyword_list.0'], df_keywords_oecds['oecds.0'])):
        tmp = []
        for keyword in keywords.split(sep=chr(0x1f)):
            if len(keyword) > 1:
                tmp.append(revers_dict_keyword[keyword])
        keywords_index.append(tmp)
        oecd_index.append(revers_dict_oecds[oecd])

    df_keywords_oecds['keywords_index'] = keywords_index
    df_keywords_oecds['oecd_index'] = oecd_index
    df_keywords_oecds.to_csv(key_words_oecds_prepared)
    #"""



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


def create_dict(df_name, dict_keywords_file, dict_oecds_file):
    """
    If function used separator: '' (ascii-code: 0x1f)
    :param df_name:
    :param dict_file:
    :return:
    """
    df_source = pd.read_csv(df_name, index_col=0)
    df_dict_keywords = pd.DataFrame()
    df_dict_oecds = pd.DataFrame()
    list_uniq_keywords = []
    for i, keyword in enumerate(df_source['keyword_list.0']):
        list_uniq_keywords += keyword.split(sep=chr(0x1f))
        #if i % 1000 == 0:
        #    print(i)

    list_uniq_keywords = sorted(list(set(list_uniq_keywords)),key=len)
    for i, keyword in enumerate(list_uniq_keywords):
        if len(keyword) == 2:
            list_uniq_keywords = sorted(list_uniq_keywords[i:])
            break

    list_uniq_oecds = sorted(list(set(df_source['oecds.0'])))
    df_dict_keywords.index.name = 'index'
    df_dict_oecds.index.name = 'index'
    df_dict_keywords['uniq_keywords'] = list_uniq_keywords
    #list_uniq_oecds += [np.NaN] * (len(list_uniq_keywords) - len(list_uniq_oecds))
    df_dict_oecds['uniq_oecds'] = list_uniq_oecds
    df_dict_keywords.to_csv(dict_keywords_file)
    df_dict_oecds.to_csv(dict_oecds_file)


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
    key_words_oecds_prepared = root_dir + "key_words_oecds_prepared.csv"
    dict_keywords_file = root_dir + "dict_keywords.csv"
    dict_oecds_file = root_dir + "dict_oecds.csv"
    ml_df_name = root_dir + "ml_df.csv"
    # create_test_csv()
    # create_new_csv(df_name, output_file)
    #prepare_csv(output_file, key_words_oecds_prepared)
    #create_dict(df_name, dict_keywords_file, dict_oecds_file)
    create_ml_df_key_words_oecds(key_words_oecds_prepared, dict_keywords_file, dict_oecds_file, ml_df_name)


if __name__ == '__main__':
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    # print(root_dir)
    logger = get_logger(__name__)
    main()
