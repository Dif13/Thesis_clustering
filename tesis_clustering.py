import sys
import os
import numpy as np
import pandas as pd
import easygui

from keras.models import load_model
MAXLEN = 20
THRESHOLD_VALUE = 0.9
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def normalization(l):
    """
    Normalizes list to one
    :param l: list with number of entries in oecds
    :return: list
    """
    s = sum(l)
    if s == 0:
        return l
    else:
        return [x / s for x in l]


def prepare_characteristic(df_dict_keywords):
    """
    Create dictionary: index - characteristic
    :param df_dict_keywords: column poligonal_characteristic
    :return: dict_characteristic
    """
    keys = df_dict_keywords.keys()
    values = df_dict_keywords.values
    for i, val in enumerate(values):
        values[i] = list(map(int, val[1:-1].split(sep=', ')))
    dict_characteristic = dict(zip(keys, values))
    return dict_characteristic


def get_tesis(input_tesis_file):
    """
    Generator. Return tesises step by step.
    :param input_tesis_file:
    :return:
    """
    with open (input_tesis_file, 'r') as file:
        # (HISTORY AND ARCHAEOLOGY),(ENGINEERING; ELECTRICAL & ELECTRONIC)
        for s_tesises in file:
            #print(s_tesises[:-1])
            l_tesises = s_tesises[:-1].upper().split(sep=', ')

            yield l_tesises


def convert(tesises, len_revers_dict_oecds, revers_dict_keyword, dict_characterictic, log_file):
    """
    Convert words to index from revers_dict_keyword (characteristics)
    :param tesises:
    :param len_revers_dict_oecds:
    :param revers_dict_keyword:
    :param dict_characterictic:
    :param log_file:
    :return:
    """
    #import pickle
    #with open('/home/diff/Документы/Projects/Python/Sochi/Thesis_clustering/X_test.pickle', 'rb') as f:
    #    X_test = pickle.load(f)
    len_tesises = len(tesises)
    num_uknown_tesises = 0
    unknown_tesises = []
    num_tesises = []
    for tesis in tesises:
        try:
            num_tesises.append(dict_characterictic[revers_dict_keyword[tesis]])
        except ValueError as e:
            print(e)
        except KeyError as e:
            num_uknown_tesises += 1
            unknown_tesises.append(tesis)

    num_tesises.sort(key=sum)
    if num_uknown_tesises != 0:
        with open(log_file, 'a') as f:
            s_tesises = ''
            for i in unknown_tesises:
                s_tesises += i + ', '
            if len(s_tesises) > 2:
                s_tesises = s_tesises[:-2]
            s = f'Unknown tesis or characteristic:' + \
                f' {num_uknown_tesises}/{len_tesises} ({round(num_uknown_tesises/len_tesises*100, 2)}%):' + \
                f' {s_tesises}'
            f.write(s + '\n')
            # print(s)

    if len(num_tesises) == 0:
        return None
    elif len(num_tesises) < MAXLEN:
        num_tesises += [[0] * len_revers_dict_oecds] * (MAXLEN - len(num_tesises))
    elif len(num_tesises) > MAXLEN:
        num_tesises = num_tesises[:MAXLEN]

    prepared_tesises = np.array([list(map(normalization, num_tesises))])
    #print(prepared_tesises.shape)
    return prepared_tesises


def convert_result(result, dict_oecds):
    """
    Convert index oecds to words
    :param result:
    :param dict_oecds:
    :return:
    """
    result_oecds = []
    result = list(result)
    #result = [list(map(r, x)) for x in result]
    for i, v in enumerate(result):
        if v > THRESHOLD_VALUE:
            result_oecds.append((dict_oecds[i], round(v, 2)))
    if len(result_oecds) == 0:
        result_oecds.append((dict_oecds[result.index(max(result))], round(max(result), 4)))
        print(f"Warning: low confidence: {result_oecds}")

    return result_oecds



def r(x):
    """
    Round functions.
    :param x:
    :return:
    """
    return round(x)


def predict(prepared_tesises, model_name):
    """
    Predict result from input data
    :param prepared_tesises:
    :param model_name:
    :return:
    """
    model = load_model(model_name)
    result = model.predict(prepared_tesises[:])

    return result[0]


def open_explorer(root_dir):
    """
    Choose csv-file with data
    :return: full-path to dataframe
    """
    input_file = easygui.fileopenbox(default=root_dir, filetypes=["*"])
    # input_file = root_dir + "dissertation.csv"
    # input_file = root_dir + "dissertation_test.csv"
    # input_file = root_dir + "key_words_oecds.csv"
    # input_file = root_dir + "key_words_oecds_prepared.csv"
    return input_file


def get_name_files():
    """
    Define names
    :return:
    """
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    dict_keywords_file = root_dir + "dict_keywords.csv"
    dict_oecds_file = root_dir + "dict_oecds.csv"
    model_name = root_dir + "models/model_1/model.11-0.831.hdf5"
    input_tesis_file = open_explorer(root_dir)
    _, output_file = os.path.split(input_tesis_file)
    output_file = 'output_' + os.path.splitext(output_file)[0] + '.csv'
    log_file = root_dir + "warning.log"

    return dict_keywords_file, dict_oecds_file, input_tesis_file, model_name, output_file, log_file


def init(dict_keywords_file, dict_oecds_file):
    """
    Download dictionary and reverse dictionary for keywords and oecds.
    Download dictionary characteristic.
    :param dict_keywords_file: dictionary keywords file
    :param dict_oecds_file: dictionary oecds file
    :return: revers_dict_keyword, revers_dict_oecds, dict_keyword, dict_oecds, dict_characterictic
    """
    df_dict_keywords = pd.read_csv(dict_keywords_file)
    df_dict_oecds = pd.read_csv(dict_oecds_file)

    revers_dict_keyword = dict(zip(df_dict_keywords['uniq_keywords'], df_dict_keywords['index']))
    revers_dict_oecds = dict(zip(df_dict_oecds['uniq_oecds'], df_dict_oecds['index']))

    dict_keyword = dict(zip(df_dict_keywords['index'], df_dict_keywords['uniq_keywords']))
    dict_oecds = dict(zip(df_dict_oecds['index'], df_dict_oecds['uniq_oecds']))

    dict_characterictic = prepare_characteristic(df_dict_keywords['polygonal_characteristic'])

    return revers_dict_keyword, revers_dict_oecds, dict_keyword, dict_oecds, dict_characterictic



if __name__ == '__main__':
    dict_keywords_file, dict_oecds_file, input_tesis_file, model_name, output_file, log_file = get_name_files()

    revers_dict_keyword, revers_dict_oecds, dict_keyword, dict_oecds, dict_characterictic = \
        init(dict_keywords_file, dict_oecds_file)

    len_revers_dict_oecds = len(revers_dict_oecds)
    get_tesises = get_tesis(input_tesis_file)
    with open(log_file, 'w') as f:
        f.write(f'Start log for {input_tesis_file}\n')

    with open(output_file, 'w') as f:
        for tesises in get_tesises:
            prepared_tesises = convert(tesises, len_revers_dict_oecds, revers_dict_keyword, dict_characterictic, log_file)

            if prepared_tesises is not None:
                rusult = predict(prepared_tesises, model_name)
                result_oecds = convert_result(rusult, dict_oecds)
            else:
                result_oecds = ['Unknown']
            s_tesises = ''
            for i in tesises:
                s_tesises += i + ', '
            if len(s_tesises) > 2:
                s_tesises = s_tesises[:-2]

            s_result_oecds = ""
            for i in result_oecds:
                s_result_oecds += str(i) + ", "
            s_result_oecds = s_result_oecds[:-2]

            f.write(str(s_tesises) + '; ' + s_result_oecds + '\n')
            #print(result_oecds


