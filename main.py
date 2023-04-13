import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit, prange, float64, int64

eps = 0.001


results_x = {}


'''
@njit(nogil=True)
def φ_n(z) -> np.array:
    x1, x2 = 6, 8
    return 4 * 16 * (np.sin((x2 - l / 2) * 2 * z / l) - np.sin((x1 - l / 2) * 2 * z / l)) / (2 * z + np.sin(2 * z))


@njit(nogil=True)
def half_method(n: int, a1: float, b1: float) -> np.ndarray:
    z = list()

    def find_c(a: float, b: float) -> float:
        G = α / (c * l)
        fz = lambda z: (np.tan(z) - G / z)
        root = (a + b) / 2
        while np.abs(a - b) > eps:
            if (fz(root) * fz(a)) < 0:
                b = root
            else:
                a = root
            root = (a + b) / 2
        return root

    while n > 0:
        root = find_c(a1, b1)
        z.append(root)
        a1 += np.pi
        b1 += np.pi
        n -= 1
    return np.array(z)


# hx = 1 / 10
x_list = np.arange(0.0, l, h_y)
print('длина листа:' + str(len(x_list)))
# ht = 1 / 10
time_list = np.arange(0.0, T, h_t)


@njit(nogil=True)
def w_n(z, φ, time=T / 2, flag='x', x=l / 2):
    def P_n(zi, φi, t):
        a2 = (k_const / c) ** 2
        aRc = (2 * α / R) / (c ** 2)

        return (φi * 4 * (1 - np.exp(-1 * ((a2 * 4 * (zi ** 2) / (l ** 2) + aRc) * t))) / (
                a2 * 4 * (zi ** 2) / (l ** 2) + aRc)) / (c * l)

    def w(z, φ, ti, x):
        s = list()
        for i in prange(len(z)):
            s.append(P_n(z[i], φ[i], ti) * np.cos(((2 * z[i] / l) * (x - l / 2))))

        return np.sum(np.array(s))

    sol = []
    if flag == 'x':
        for i in x_list:
            sol.append(w(z, φ, time, i))
    elif flag == 't':
        for i in time_list:
            sol.append(w(z, φ, i, x))

    return sol


@njit(nogil=True)
def solutions(n, t=T, flag='x', x=l / 2):
    z = half_method(n, 0.00001, np.pi / 2)
    φ = φ_n(z)
    solution = w_n(z, φ, t, flag=flag, x=x)

    return solution
'''


def k_steps(I: int):
    return int((2 * T * I ** 2) // (c * l ** 2))


def solver_explicit_simple(I, α, c, l, T, k_const, R):
    """
    Простейшая реализация явной разностной схемы для приближенного решения
    параболического уравнения.
    """

    K = k_steps(I)
    print(K)
    h_y = l / I
    h_t = T / K

    print(h_y)
    φ_y = np.zeros(I)
    for i in range(int(l / (3 * h_y)), int(2 * l / (3 * h_y))):
        φ_y[i] = 16
    y = np.linspace(0, l, I)
    t = np.linspace(0, T, K)
    w = np.zeros((I, K))
    for k in range(0, K - 1):
        # Вычисляем приближенное решение во внутренних узлах сетки
        for i in range(1, I - 1):
            w[i, k + 1] = w[i, k] + (k_const * h_t / (c * h_y * h_y)) * (w[i + 1, k] - 2 * w[i, k] + w[i - 1, k]) - (
                        2 * h_t * α / (R / 2 * c * c)) * w[i, k] + h_t * φ_y[i] / c

        # Задаем граничные условия
        # w[0, k + 1] = k_const * h_t / (c * h_y**2) * (2 * w[1, k] - 2 * w[0, k]) + 2 * h_t * α / (R*c**2) * w[0, k] + h_t * φ_y[i]
        # w[i, k + 1] = k_const * h_t / (c * h_y**2) * (w[i - 1, k] - w[i, k]) + 2 * h_t * α / (R*c**2) * w[i, k]
        w[0, k + 1] = k_const * h_t * (2 * w[1, k] - 2 * w[0, k]) / (c * h_y ** 2) + (
                    1 - (h_t * 2 * α) / (R * c ** 2)) * w[0, k] + φ_y[0]
        w[I - 1, k + 1] = k_const * h_t * (2 * w[i - 1, k] - 2 * h_y * (α / c) * w[i, k] - 2 * w[i, k]) / (c * h_y ** 2) + w[i, k] - (
                    h_t * 2 * α) / (R * c ** 2) * w[i, k] + h_t * φ_y[i] / c

    # ================================================

    # results_x[250] = solutions(250, 250)  # аналитическое решение в момент времени 250 сек
    # print(results_x[125][0])
    # print('===============' + str(len(results_x[250])))
    # print('узел в аналитика: ' + str(results_x[250][49]))

    # ================================================
    plt.subplot(2, 1, 1)
    #  plt.plot(y, results_x[250], label=str(int(T)) + ' C аналитика')
    w_l = w[:, int(K-1)]
    plt.plot(y, w_l, label=str(int(T)) + ' C')
    w_l = w[:, int(200 / h_t)]
    plt.plot(y, w_l, label=str(int(200)) + ' C')
    w_l = w[:, int(150 / h_t)]
    plt.plot(y, w_l, label=str(int(150))+' C')
    w_l = w[:, int(100 / h_t)]
    plt.plot(y, w_l, label=str(int(100)) + ' C')
    w_l = w[:, int(50 / h_t)]
    plt.plot(y, w_l, label=str(int(50)) + ' C')
    w_l = w[:, int(25 / h_t)]
    plt.plot(y, w_l, label=str(int(25)) + ' C')
    plt.grid()
    plt.legend()
    plt.xlabel("x, CM")
    plt.ylabel("w, K")
    # ==========================================
    plt.subplot(2, 1, 2)
    w_t = w[int(I / 2), :]
    plt.plot(t, w_t, label=str(int(l / 2)) + ' CM')
    w_t = w[int(I / 3), :]
    plt.plot(t, w_t, label=str(int(l / 3)) + ' CM')
    w_t = w[int(I / 4), :]
    plt.plot(t, w_t, label=str(int(l / 4)) + ' CM')
    w_t = w[int(I / 5), :]
    plt.plot(t, w_t, label=str((l / 5)) + ' CM')
    w_t = w[int(I / 6), :]
    plt.plot(t, w_t, label=str((l / 6)) + ' CM')
    plt.grid()
    plt.legend()
    plt.xlabel("t, C")
    plt.ylabel("w, K")

    plt.show()


def solver_explicit_simple_epsilon(I, α, c, l, T, K, k_const, R, node: int):
    # node должен быть в диапазоне от 0 до I - 1
    h_y = l / I
    h_t = T / K
    # print(I, K)
    φ_y = np.zeros(I)
    for i in range(int(l / (3 * h_y)), int(2 * l / (3 * h_y))):
        φ_y[i] = 16
    w = np.zeros((I, K))
    for k in range(1, K - 1):
        # Вычисляем приближенное решение во внутренних узлах сетки
        for i in range(1, I - 1):
            w[i, k + 1] = w[i, k] + (k_const * h_t / (c * h_y ** 2)) * (w[i + 1, k] - 2 * w[i, k] + w[i - 1, k]) - (
                    2 * h_t * α / (R / 2 * c ** 2)) * w[i, k] + h_t * φ_y[i] / c
            # Граничные условия
        w[0, k + 1] = k_const * h_t * (2 * w[1, k] - 2 * w[0, k]) / (c * h_y ** 2) + (1 - (h_t * 2 * α) / (R * c ** 2)) * w[0, k] + φ_y[0]
        w[I - 1, k + 1] = k_const * h_t * (2 * w[i - 1, k] - 2 * h_y * (α / c) * w[i, k] - 2 * w[i, k]) + w[i, k] - (h_t * 2 * α) / (R * c ** 2) * w[i, k] + h_t * φ_y[i] / c
    return w[node, int(K-1)]


def epsilon(l_steps: int, node: int):
    α = 0.001  # Вт/(см^2*град)
    c = 1.65  # Дж/(cм^3*град)
    l = 12  # см
    T = 250  # с
    k_const = 0.59  # Вт/(см*град)
    R = 0.1  # Радиус стержня
    # results_x[125] = solutions(125, 125)  # аналитическое решение в момент времени 125 сек
    # print(results_x[125][0])
    # print('узел в аналитике: ' + str(results_x[125][int(I_steps/2)]))
    # result_x = results_x[125][int(I_steps/2)]
    # l_steps = 50
    k_eps = k_steps(l_steps)
    result_x = solver_explicit_simple_epsilon(l_steps, α, c, l, T, k_eps, k_const, R, node)
    # print(result_x)
    print(f"узел = {node} при I = {l_steps}: " + str(result_x))
    l_steps = int(l_steps * 2)
    node = int(node * 2)
    k_eps = k_steps(l_steps)
    infelicity1 = solver_explicit_simple_epsilon(l_steps, α, c, l, T, k_eps, k_const, R, node)
    # print(infelicity1)
    infelicity = np.abs(result_x - infelicity1)
    print(f"узел = {node} при I = {l_steps}: " + str(infelicity1))
    l_steps = int(l_steps * 2)
    node = int(node * 2)
    k_eps = k_steps(l_steps)
    infelicity2 = solver_explicit_simple_epsilon(l_steps, α, c, l, T, k_eps, k_const, R, node)
    # print(infelicity2)
    print(f"узел = {node} при I = {l_steps}: " + str(infelicity2))
    infelicity2 = np.abs(infelicity - infelicity2)

    infelicity = infelicity1 / infelicity2
    print('delta= ' + str(infelicity))
    print('=======================================')


if __name__ == '__main__':
    I_steps = 100  # кол-во отсчётов I
    # K = 20000  # кол-во отсчётов K
    α = 0.001  # Вт/(см^2*град)
    c = 1.65  # Дж/(cм^3*град)
    l = 12  # см
    T = 250  # с
    k_const = 0.59  # Вт/(см*град)
    R = 0.1  # Радиус стержня
    # solver_explicit_simple(I_steps, α, c, l, T, k_const, R)
    epsilon(10, 5)
    epsilon(10, 4)
    epsilon(10, 3)
    epsilon(10, 2)

    epsilon(50, 25)
    epsilon(50, 20)
    epsilon(50, 15)
    epsilon(50, 10)

