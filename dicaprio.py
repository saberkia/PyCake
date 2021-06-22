import pandas as pd
import numpy as np
import pickle


def load_txt(filename):
    with open(filename, mode='r') as file:
        text = file.read()
    return text


def load_txt_np(filename, delimiter=',', dtype=str):
    return np.loadtxt(filename, delimiter=delimiter, dtype=dtype)


def load_csv(filename, delimiter=',', dtype=None):
    return pd.read_csv(filename, delimiter=delimiter, dtype=dtype)


def load_tsv(filename, delimiter='\t', dtype=None):
    return pd.read_csv(filename, delimiter=delimiter, dtype=dtype)


def load_csv_np(filename):
    return np.recfromcsv(filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


def load_excel(filename):
    return pd.ExcelFile(filename, engine='openpyxl').parse(1)


if __name__ == "__main__":
    print('hello world !')
    xls = load_excel('Data of khakzad.xlsx')
    print(xls.head())
    print(xls.columns)
