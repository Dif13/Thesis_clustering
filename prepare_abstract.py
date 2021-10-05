import numpy as np
import pandas as pd
import sys
import os
import logging
import math

NUMBER_CHAR = 1000


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
    #def_logger.addHandler(get_file_handler())
    def_logger.addHandler(get_stream_handler())
    return def_logger


def get_names():
    names = {
        'input_test_file': root_dir + "dissertation_test.csv",
        'input_file': root_dir + "dissertation.csv",
        'all_set_char_file': root_dir + "all_set_char.txt",
        'dict_oecds_file': root_dir + "dict_oecds.csv"
    }
    return names


def get_column(input_file, column):
    df = pd.read_csv(input_file)
    return df[column]


def show_plot(distribution):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Диаграмма количества ключевых слов')
    ax.bar(distribution.keys(), distribution.values())

    ax.set_facecolor('seashell')
    fig.set_facecolor('floralwhite')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure
    ax.set_xlabel('Длинна текста')
    ax.set_ylabel('Частота использования ключевых слов')

    plt.show()


def len_split(x) -> int:
    try:
        return len(x.split(sep=', '))#chr(0x1f)))

    except AttributeError:
        return 0


def show_distribution(df_abstract, operatinon=len):
    distribution = {}
    for i, text in enumerate(df_abstract):
        try:
            l = operatinon(text)
        except TypeError:
            print(i, text)
        if l in distribution:
            distribution[l] += 1
        else:
            distribution[l] = 1

    print(max(distribution.keys()))
    print(distribution)
    show_plot(distribution)


def create_char_dict(set_char_filename, df_abstract):
    all_set_char = set()
    print(len(df_abstract))
    for text in df_abstract:
        try:
            all_set_char = all_set_char.union(set(text))
            # all_set_char = all_set_char.union(set(text.split()))
        except TypeError:
            continue
        except AttributeError:
            continue

    all_set_char = sorted(list(set(str(all_set_char).lower())))
    str_char = ''
    for char in all_set_char:
        str_char += f'{ord(char)} - ' + char + '\n' #chr(0x1f)
    #str_char = str_char[:-1]
    with open(set_char_filename, 'w') as f:
        f.write(str_char)
    # print(all_set_char)
    # print(f"len(all_set_char): {len(all_set_char)}")
    return all_set_char


def prepare_data(tuple_char, abstract):
    """
    Create list with index char in tuple_char.
    :param tuple_char:
    :param abstract:
    :return: indexes
    """
    indexes = []
    try:
        for char in abstract.lower():
            try:
                ind = tuple_char[0].index(char)
            except ValueError:
                if char in tuple_char[1]:
                    ind = len(tuple_char)
                elif char in tuple_char[2]:
                    ind = len(tuple_char) + 1
                else:
                    # logger.info(f'special symbol: {char}')
                    ind = None
            indexes.append(ind)
    except AttributeError :
        logger.warning(f'NaN in abstract: {abstract}')
        pass

    if len(indexes) > NUMBER_CHAR:
        indexes = indexes[:1000]
    # print(indexes)
    # print(len(indexes))
    return indexes


def quantization_abstract(indexes, tuple_char):
    quantum_abstract = np.zeros((NUMBER_CHAR, (len(tuple_char[0]) + len(tuple_char) - 1)), bool)
    for i, ind in enumerate(indexes):
        if ind:
            quantum_abstract[i][ind] = True
    return quantum_abstract


def quantization_oecds(oecds, dict_oecds):
    quantum_oecds = np.zeros(len(dict_oecds), bool)
    try:
        splited_oecds = oecds.split(sep=', ')
        for oecd in splited_oecds:
            quantum_oecds[dict_oecds[oecd.lower()]] = True
    except AttributeError:
        logger.warning(f'NaN in oecds')
        raise(f'NaN in oecds')
    return quantum_oecds



def get_tuple_char():
    """
    Tuple_char is tuple ((alphabetical), (upper), (lower), (latin))
    alphabetical - is real index. (tuple_char.index(char))
    (upper), (lower), (latin) - len(tuple_char), len(tuple_char) + 1,len(tuple_char) + 2.
    :return:
    """
    tuple_char = (
        (' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>',
        '?', '@', '[', '\\', ']', '^', '_', '`',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        '{', '|', '}', '~',
        'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о',
        'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э',
        'ю', 'я', 'ё'),
        ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁺', '⁻', '⁼', '⁽', '⁾', 'ⁿ'),
        ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₊', '₋', '₌', '₍', '₎', 'ₒ', 'ₓ', 'ₙ'),
        ('α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο',
         'π', 'ρ', 'ς', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω')
    )

    return tuple_char


def create_dict_oecds(df_oecds, dict_oecds_file):
    list_oecd = []
    for oecds in df_oecds:
        #print(oecds)
        try:
            splited_oecds = oecds.split(sep=', ')
            for oecd in splited_oecds:
                list_oecd.append(oecd.lower())
        except AttributeError:
            logger.warning(f'NaN in oecds')

    list_oecd = sorted(list(set(list_oecd)))
    print(list_oecd)
    print(len(list_oecd))

    df_dict_oecds = pd.DataFrame()
    df_dict_oecds.index.name = 'index'
    df_dict_oecds['uniq_oecds'] = list_oecd
    df_dict_oecds.to_csv(dict_oecds_file)
    logger.info(f"Done!")
    dict_oecds = {}
    for i, oecd in enumerate(list_oecd):
        dict_oecds[oecd] = i
    return dict_oecds

def main():
    names = get_names()
    # df_abstract_oecds = get_column(names['input_test_file'], ['abstract', 'oecds.0'])
    df_abstract_oecds = get_column(names['input_file'], ['abstract', 'oecds.0'])
    # show_distribution(df_abstract)
    #all_set_char = create_char_dict(names['all_set_char_file'], df_abstract)
    tuple_char = get_tuple_char()
    quantum_abstracts = []
    quantum_oecds = []
    len_df_abstract = len(df_abstract_oecds['abstract'])
    dict_oecds = create_dict_oecds(df_abstract_oecds['oecds.0'], names['dict_oecds_file'])
    #get_oecds_tokinizer(df_abstract_oecds['oecds.0'])
    for i, (abstract, oecds) in enumerate(zip(df_abstract_oecds['abstract'], df_abstract_oecds['oecds.0'])):

        if i%1000 == 0:
            print(f'{i}/{len_df_abstract}')

        if abstract is np.NaN or oecds is np.NaN:
            # list_keyword.append(np.NaN)
            continue

        indexes = prepare_data(tuple_char, abstract)
        quantum_abstracts.append(quantization_abstract(indexes, tuple_char))
        quantum_oecds.append(quantization_oecds(oecds, dict_oecds))
        #sys.exit()

    # df_oecds = get_column(input_file, 'oecds.0')
    # show_distribution(df_oecds, len_split)



if __name__ == '__main__':
    root_dir = os.path.split(sys.argv[0])[0] + '/'
    logger = get_logger(__name__)
    main()
