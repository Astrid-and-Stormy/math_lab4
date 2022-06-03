import math
import numpy as np
import matplotlib.pyplot as plt


def f1(x):
    return (5 * x) / (x ** 4 + 7)


def f2(x):
    return (3*x)/(x**4+1)


def f3(x):
    return (6*x)/(x**4+3)


def lineal_function(x, a, b):
    return x * a + b


def lineal_approximation(x, y, n):
    SX = sum(x)
    SY = sum(y)
    SXX = 0
    SXY = 0
    for i in range(n):
        SXX += x[i] * x[i]
        SXY += x[i] * y[i]
    delta = SXX * n - SX * SX
    delta1 = SXY * n - SX * SY
    delta2 = SXX * SY - SX * SXY

    return delta1 / delta, delta2 / delta


def square_function(x, a, b, c):
    return a * x ** 2 + b * x + c


def square_approximation(x, y, n):
    SX = sum(x)
    SY = sum(y)
    SXX = SXXX = SXXXX = SXY = SXXY = 0
    for i in range(n):
        SXX += x[i] * x[i]
        SXXX += x[i] ** 3
        SXXXX += x[i] ** 4
        SXY += x[i] * y[i]
        SXXY += x[i] * x[i] * y[i]
    delta0 = np.linalg.det([
        [n, SX, SXX],
        [SX, SXX, SXXX],
        [SXX, SXXX, SXXXX]])
    delta1 = np.linalg.det([
        [SY, SX, SXX],
        [SXY, SXX, SXXX],
        [SXXY, SXXX, SXXXX]]
    )
    delta2 = np.linalg.det([
        [n, SY, SXX],
        [SX, SXY, SXXX],
        [SXX, SXXY, SXXXX]]
    )
    delta3 = np.linalg.det([
        [n, SX, SY],
        [SX, SXX, SXY],
        [SXX, SXXX, SXXY]]
    )
    return delta3 / delta0, delta2 / delta0, delta1 / delta0


def triple_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def triple_approximation(x, y, n):
    data = [[0] * 5 for _ in range(4)]
    for i in range(4):
        for j in range(n):
            for k in range(4):
                data[i][k] += x[j] ** (i + k)
            data[i][4] += x[j] ** i * y[j]
    data0 = [
        [data[0][0], data[0][1], data[0][2], data[0][3]],
        [data[1][0], data[1][1], data[1][2], data[1][3]],
        [data[2][0], data[2][1], data[2][2], data[2][3]],
        [data[3][0], data[3][1], data[3][2], data[3][3]]
    ]
    data1 = [
        [data[0][4], data[0][1], data[0][2], data[0][3]],
        [data[1][4], data[1][1], data[1][2], data[1][3]],
        [data[2][4], data[2][1], data[2][2], data[2][3]],
        [data[3][4], data[3][1], data[3][2], data[3][3]]
    ]
    data2 = [
        [data[0][0], data[0][4], data[0][2], data[0][3]],
        [data[1][0], data[1][4], data[1][2], data[1][3]],
        [data[2][0], data[2][4], data[2][2], data[2][3]],
        [data[3][0], data[3][4], data[3][2], data[3][3]]
    ]
    data3 = [
        [data[0][0], data[0][1], data[0][4], data[0][3]],
        [data[1][0], data[1][1], data[1][4], data[1][3]],
        [data[2][0], data[2][1], data[2][4], data[2][3]],
        [data[3][0], data[3][1], data[3][4], data[3][3]]
    ]
    data4 = [
        [data[0][0], data[0][1], data[0][2], data[0][4]],
        [data[1][0], data[1][1], data[1][2], data[1][4]],
        [data[2][0], data[2][1], data[2][2], data[2][4]],
        [data[3][0], data[3][1], data[3][2], data[3][4]]
    ]
    delta0 = np.linalg.det(data0)
    delta1 = np.linalg.det(data1)
    delta2 = np.linalg.det(data2)
    delta3 = np.linalg.det(data3)
    delta4 = np.linalg.det(data4)
    return delta4 / delta0, delta3 / delta0, delta2 / delta0, delta1 / delta0


def exponential_function(x, a, b):
    return a * math.e ** (b * x)


def exponential_approximation(x, y, n):
    y1 = y.copy()
    is_log = True
    for i in range(n):
        try:
            y1[i] = math.log(y[i], math.e)
        except:
            is_log = False
            break
    if is_log:
        a1, a0 = lineal_approximation(x, y1, n)
        a = math.e ** a0
        return a, a1
    return None, None


def logarithmic_function(x, a, b):
    return a * math.log(x, math.e) + b


def logarithmic_approximation(x, y, n):
    x1 = x.copy()
    is_log = True
    for i in range(n):
        try:
            x1[i] = math.log(x[i], math.e)
        except:
            is_log = False
            break
    if is_log:
        a1, a0 = lineal_approximation(x1, y, n)
        return a1, a0
    return None, None


def power_function(x, a, b):
    return a * x ** b


def power_approximation(x, y, n):
    x1 = x.copy()
    y1 = y.copy()
    is_log = True
    for i in range(n):
        try:
            x1[i] = math.log(x[i], math.e)
            y1[i] = math.log(y[i], math.e)
        except:
            is_log = False
            break
    if is_log:
        a1, a0 = lineal_approximation(x1, y1, n)
        return math.e ** a0, a1
    return None, None


def found_square_error(func1, func2, n):
    summa = 0
    for i in range(n):
        summa += (func1[i] - func2[i]) ** 2
    return summa


def approximation(x, y, n):
    lf = []
    sf = []
    tf = []
    lgf = []
    ef = []
    pf = []
    la, lb = (lineal_approximation(x, y, n))
    sa, sb, sc = (square_approximation(x, y, n))
    ta, tb, tc, td = (triple_approximation(x, y, n))
    is_log = True
    lga, lgb = (logarithmic_approximation(x, y, n))
    if lga is None:
        is_log = False
    is_exp = True
    ea, eb = (exponential_approximation(x, y, n))
    if ea is None:
        is_exp = False
    is_pow = True
    pa, pb = (power_approximation(x, y, n))
    if pa is None:
        is_pow = False
    for i in range(n):
        lf.append(lineal_function(x[i], la, lb))
        sf.append(square_function(x[i], sa, sb, sc))
        tf.append(triple_function(x[i], ta, tb, tc, td))
        if is_log:
            lgf.append(logarithmic_function(x[i], lga, lgb))
        if is_exp:
            ef.append(exponential_function(x[i], ea, eb))
        if is_pow:
            pf.append(power_function(x[i], pa, pb))
    s_error = [found_square_error(lf, y, n), found_square_error(sf, y, n), found_square_error(tf, y, n)]
    if is_log:
        s_error.append(found_square_error(lgf, y, n))
    if is_exp:
        s_error.append(found_square_error(ef, y, n))
    if is_pow:
        s_error.append(found_square_error(pf, y, n))
    square_error = [0] * len(s_error)
    for i in range(len(s_error)):
        square_error[i] = (s_error[i] / n) ** 0.5
    Name_of_functions = ["линейная функция", "полиномиальная функция 2-ой степени",
                         "полиномиальная функция 3-ей степени", "экспоненциальная функция", "логарифмическая функция",
                         "степенная функция"]
    print("Лучшая апроксимирующая функция -", Name_of_functions[square_error.index(min(square_error))])
    plt.plot(x, y, 'o')
    x_nump = np.arange(x[0], x[-1], 0.01)
    plt.plot(x_nump, la * x_nump + lb)
    plt.plot(x_nump, sa * x_nump ** 2 + sb * x_nump + sc)
    plt.plot(x_nump, ta * x_nump ** 3 + tb * x_nump ** 2 + tc * x_nump + td)
    if is_log:
        x_log = []
        for i in x_nump:
            x_log.append(math.log(i, math.e))
        x_log = np.array(x_log)
        plt.plot(x_nump, lga * x_log + lgb)
    if is_exp:
        plt.plot(x_nump, ea * math.e ** (eb * x_nump))
    if is_pow:
        plt.plot(x_nump, pa * x_nump ** pb)
    plt.show()


# x = [1.1, 2.3, 3.7, 4.5, 5.4, 6.8, 7.5]
# y = [2.73, 5.12, 7.74, 8.91, 10.59, 12.75, 13.43]


def get_x_and_y(a, b, h):
    x = []
    y = []
    for i in range(int((b - a) / h) + 1):
        x.append(a)
        y.append(f1(x[i]))
        a += h
    return x, y


# x, y = get_x_and_y(-2, 0, 0.2)
# approximation(x, y, len(x))


def read_from_cons(n):
    x = []
    y = []
    print("Введите x и y через пробел. Каждую пару значений на новой строчке")
    rows = 0
    while n != rows:
        try:
            row = list(map(float, input().split()))
        except:
            print("Неверный формат")
            continue

        if len(row) != 2:
            print("Неверное количество чисел")
            continue

        rows += 1
        x.append(row[0])
        y.append(row[1])
    return x, y, n


def read_from_file(f):
    x = []
    y = []
    n = 0
    text = f.read()
    accuracy = -1
    for line in text.split("\n"):
        try:
            row = list(map(float, line.split()))
        except:
            print("Данные в файле неправильные")
            return None, None, None
        if len(row) == 2:
            x.append(row[0])
            y.append(row[1])
            n += 1
        else:
            print("Неверное количество переменных")
            return None, None, None
    return x, y, n


def read_func():
    print("Выберите функцию\n"
          "1. 5x/(x^4+7)\n"
          "2. 3x/(x^4+1)\n"
          "3. 6x/(x^4+3)")
    try:
        n = int(input())
    except:
        print("Введите число")
        return None, None, None
    if not (1 <= n <= 3):
        print("Ты че угараешь? Тут всего 3 цифры!")
        return None, None, None
    n -= 1
    functions = [f1, f2, f3]
    x = [0.2]
    y = [functions[n](0.2)]
    for i in range(9):
        x.append(x[i]+0.2)
        y.append(functions[n](x[i+1]))
    return x, y, 10


def read_data():
    print("Введите имя файла или количество переменных. Или выберете одну из предложенных функций, введя func. "
          "Чтобы выйти, отправьте exit")
    inp = input()
    if inp.isnumeric():
        inp = int(inp)
        if inp > 1:
            x, y, n = read_from_cons(inp)
        elif inp == 1:
            print("И как ты себе это представляешь?")
            x, y, n = read_data()
        else:
            print("Введите положительное число")
            x, y, n = read_data()
    elif inp == "exit":
        x, y, n = None, None, -1
    elif inp == "func":
        x, y, n = read_func()
    else:
        try:
            with open(inp, "r+") as f:
                x, y, n = read_from_file(f)
        except:
            print("Ошибка чтения файла")
            x, y, n = read_data()
    return x, y, n


def main():
    print("Добро пожаловать! Давайте найдем функцию, которая более менее удовлетворяет нашим данным")
    while True:
        x, y, n = read_data()
        while n is None:
            x, y, n = read_data()
        if n == -1:
            print("До свидания. Ждем вас еще")
            break
        print("Закройте график, чтобы продолжить")
        approximation(x, y, n)


main()
