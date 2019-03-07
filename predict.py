# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """

    w1 = np.loadtxt('w1.txt')
    w2 = np.loadtxt('w2.txt')


    bias1 = np.ones((x.shape[0], 1))
    x_bias = np.concatenate((bias1, x), axis=1)
    output1 = sigmoid(w1 @ np.transpose(x_bias))
    bias2 = np.ones((1, output1.shape[1]))
    output1 = np.concatenate((bias2, output1), axis=0)
    output2 = w2 @ output1

    y_pred = np.argmax(output2, axis=0)

    print(y_pred.shape)
    print(type(y_pred))

    y_pred=np.reshape(y_pred,((x.shape[0]),1))

    return y_pred




def sigmoid(x):
    result=1/(1+np.exp(-x))

    return result

