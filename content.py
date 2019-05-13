# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np
from functools import partial


def sig(x):
    return 1/(1+np.exp(-x))


sigm = np.vectorize(sig)


def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return sigm(x)


def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    N, M = np.shape(x_train)
    f = sigmoid(x_train@w)

    log = np.divide(y_train * np.log(f) + (1 - y_train) * np.log(1 - f), -1 * N)

    f_sub_y = f - y_train
    sum_on_N = np.sum(f_sub_y) / N
    gradient = x_train.T @ f_sub_y

    grad = np.sum(gradient, axis=1) / N
    grad = np.reshape(grad, (M, 1))

    """
    print("\n\n\n\n############# HERE")
    print(np.shape(f))
    print(np.shape(y_train))
    print(np.shape(gradient))
    print((N, M))
    """
    return np.sum(log), grad


def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """

    # theta
    w = w0
    log_values = []
    f, grad = obj_fun(w)
    log_values.append(f)
    for i in range(0, epochs):

        w = w - eta*grad
        log_values.append(f)
        f, grad = obj_fun(w)
    log_values.append(f)

    # dafuq
    del log_values[0:2]

    # np.reshape(log_values, (epochs, 1))
    return w, log_values


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """

    M, _ = np.shape(y_train)
    m = int(M/mini_batch)

    x_mini_batch = np.vsplit(x_train, m)
    y_mini_batch = np.vsplit(y_train, m)

    w = w0
    log_values = []
    for i in range(0, epochs):
        for j in range(0, m):
            _, grad = obj_fun(w, x_mini_batch[j], y_mini_batch[j])
            w = w - eta*grad
        f, _ = obj_fun(w, x_train, y_train)
        log_values.append(f)

    # np.reshape(log_values, (epochs, 1))
    return w, log_values


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """

    log, grad = logistic_cost_function(w, x_train, y_train)
    w = np.array(w)
    w[0] = 0
    log += np.sum(np.square(w)) * regularization_lambda / 2
    grad += (regularization_lambda * w)
    return log, grad


def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    def f(x): return sigmoid(x) >= theta
    f = np.vectorize(f)
    return f(x@w)


def f_measure(y_true, y_pred):
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    TP = np.sum(np.logical_and(y_true, y_pred))
    FP_PN = np.sum(np.logical_xor(y_true, y_pred))
    return 2*TP/(2*TP+FP_PN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """

    N, M = np.shape(x_train)
    len_lambdas = len(lambdas)
    len_thetas = len(thetas)

    F = np.zeros((len_lambdas, len_thetas))
    w = w0
    best_w = 0
    best_theta = 0
    best_lambda = 0
    best_f = -np.inf

    for i, lambda_ in enumerate(lambdas):
        for j, theta in enumerate(thetas):
            def obj_funct(w, x, y): return regularized_logistic_cost_function(w, x, y, lambda_)
            w, _ = stochastic_gradient_descent(
                obj_funct, x_train, y_train, w0, epochs, eta, mini_batch)
            f = f_measure(y_val, prediction(x_val, w, theta))
            F[i, j] = f
            if f > best_f:
                best_w = w
                best_theta = theta
                best_lambda = lambda_
                best_f = f
    return best_lambda, best_theta, best_w, F
