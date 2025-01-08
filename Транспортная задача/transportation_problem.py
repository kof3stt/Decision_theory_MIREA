import csv
import sympy
import copy
import re
import math
import numpy as np
from itertools import product


def transport_task():
    '''Функция, отвечающая за решение закрытой транспортной задачи'''
    # поставщики, потребители, c = input_data()
    поставщики, потребители, c = input_data_from_file()
    assert sum(поставщики) == sum(
        потребители), 'Транспортная задача не является закрытой'
    C = np.vstack(c)  # Стоимости перевозок единицы груза из Ai в Bi
    X = np.zeros_like(C)
    basis = list()  # Базисные переменные (заполненные клетки)
    U = np.zeros_like(поставщики)  # Потенциалы пунктов Ai
    V = np.zeros_like(потребители)  # Потенциалы пунктов Bj
    delta = np.zeros_like(C)  # Относительные оценки клеток
    marks = list(product([f'u{i}' for i in range(
        1, len(поставщики) + 1)], [f'v{i}' for i in range(1, len(потребители) + 1)]))
    num_iteration = 0  # Номер итерации в методе потенциалов

    def print_table(potential=True):
        '''Функция для вывода таблицы'''
        print('-' * (14 * (len(потребители) + 2) + len(потребители) + 2))
        if potential:
            print(' ' * 14, end='|')
            for i in range(len(потребители)):
                print(f'v{i + 1} = {V[i]}'.ljust(14), end='|')
            print(''.ljust(14), end='|')
            print('\n' + ' ' * 14 + ('|' + '-' * 14)
                  * len(потребители), end='|')
            print(''.ljust(14), end='|')
            print()
        print('Пункты'.ljust(14), end='|')
        for i in range(len(потребители)):
            print(f'B{i + 1}'.ljust(14), end='|')
        print('Запасы'.ljust(14), end='|')
        for i in range(len(поставщики)):
            print('\n' + '-' * (14 * (len(потребители) + 2) + len(потребители) + 2))
            if potential:
                print(f'u{i + 1} = {U[i]}'.ljust(14), end='|')
            else:
                print(f'A{i + 1}'.ljust(14), end='|')
            for j in range(len(потребители)):
                print(f'{C[i][j]}'.rjust(14), end='|')
            print('\033[94m' + f'{поставщики[i]
                                  }'.ljust(14) + '\033[0m', end='|')
            if potential:
                print('\n' + f'A{i + 1}'.ljust(14), end='|')
            else:
                print('\n' + ' ' * 14, end='|')
            for j in range(len(потребители)):
                if (i, j) not in basis:
                    print(f''.ljust(14), end='|')
                else:
                    print('\033[102m' +
                          f'{X[i][j]}'.ljust(14) + '\033[0m', end='|')
            print(''.ljust(14), end='|')
        print('\n' + '-' * (14 * (len(потребители) + 2) + len(потребители) + 2))
        print('Потребности'.ljust(14), end='|')
        for i in range(len(потребители)):
            print('\033[94m' + f'{потребители[i]
                                  }'.ljust(14) + '\033[0m', end='|')
        print('\033[95m' + f'{sum(потребители)
                              }'.ljust(14) + '\033[0m', end='|')
        print('\n' + '-' * (14 * (len(потребители) + 2) +
              len(потребители) + 2))

    def northwest_corner_method():
        '''Метод северо-западного угла нахождения начального опорного решения'''
        for i in range(len(поставщики)):
            for j in range(len(потребители)):
                X[i][j] = min(поставщики[i] - np.sum(X[i, :]),
                              потребители[j] - np.sum(X[:, j]))
                if X[i][j] != 0:
                    basis.append((i, j))
        if len(basis) < len(поставщики) + len(потребители) -1:
            basis.append((2,2))


    def get_min_indexes(suppliers, consumers):
        '''Вспомогательная функция для определения индексов ячейки с минимальной стоимостью'''
        min_cost = np.inf  # хранит наименьший элемент, который нашли на текущий момент в матрице С
        min_indexes = (None, None)  # индексы наименьшего элемента
        for i in range(len(C)):
            for j in range(len(C[0])):
                if min_cost > C[i][j] and C[i][j] > 0 and X[i][j] == 0:
                    if suppliers[i] > 0 and consumers[j] > 0:
                        # назначаем (i, j) элемент новым наименьшим
                        min_cost = C[i][j]
                        min_indexes = (i, j)
        return min_indexes

    def min_price_method():
        suppliers = np.copy(поставщики)  # Копия вектора поставщиков
        consumers = np.copy(потребители)  # Копия вектора потребителей
        '''Метод минимальной стоимости нахождения опорного решения'''
        while True:
            i, j = get_min_indexes(suppliers, consumers)
            if i is None and j is None:
                # все потребности удовлетворены и/или все возможности использованы
                break
            resources = min(suppliers[i], consumers[j])
            suppliers[i] = suppliers[i] - resources
            consumers[j] = consumers[j] - resources
            X[i][j] = resources
            basis.append((i, j))

    def count_function():
        '''Расчёт стоимости перевозок по текущему плану'''
        return np.dot(C.reshape(len(поставщики) * len(потребители)), X.reshape(len(поставщики) * len(потребители)))

    def count_potentials(calculation_output=False):
        ''' Функция для подсчёта потенциалов Ui, Vi'''
        system = []
        for u, v in marks:
            if (int(u[1]) - 1, int(v[1]) - 1) in basis:
                system.append(f'{u} + {v} = {C[int(u[1]) - 1][int(v[1]) - 1]}')
        if calculation_output:
            print("Вычислим потенциалы ui и vi, исходя из базисных переменных. Для их нахождения используем условия ui + vj = cij")
            print(*system, sep='\n')
        first_equation = system[0]
        first_variable = first_equation.split()[0]
        variables = set()
        for equation in system:
            for symbol in equation.split():
                if re.fullmatch(r'[uv]\d*', symbol):
                    variables.add(symbol)
        symbols_dict = {symbol: sympy.symbols(symbol) for symbol in variables}
        system[0] = first_equation.replace(first_variable, '0')
        equations = []
        for equation in system:
            left, right = equation.split('=')
            equations.append(
                sympy.Eq(sympy.sympify(left), sympy.sympify(right)))
        equations.append(sympy.Eq(symbols_dict[first_variable], 0))
        solution = sympy.solve(equations, list(symbols_dict.values()))
        for key, value in solution.items():
            if str(key)[0] == 'u':
                U[int(str(key)[1]) - 1] = value
            else:
                V[int(str(key)[1]) - 1] = value
        if calculation_output:
            print(f'Считая, что {first_variable} = 0, имеем:')
            print(*[f'{key} = {value}' for key,
                  value in solution.items()], sep='; ')

    def count_delta(calculation_output=False):
        nonlocal delta
        if calculation_output:
            print('Для каждой свободной клетки вычислим относительные оценки:')
        delta = np.zeros_like(C)
        for i in range(len(поставщики)):
            for j in range(len(потребители)):
                if (i, j) not in basis:
                    delta[i][j] = C[i][j] - (U[i] + V[j])
                    if count_delta:
                        print(f'Δ{i + 1}{j + 1} = {delta[i][j]};')

    def recalculate_optimal_plan():
        '''Функия перерасчёта оптимального плана'''
        min_i, min_j = math.inf, math.inf
        lowest_mark = math.inf
        for i in range(len(поставщики)):
            for j in range(len(потребители)):
                if delta[i][j] < lowest_mark:
                    lowest_mark = delta[i][j]
                    min_i = i
                    min_j = j
        basis.append((min_i, min_j))
        available_ways = copy.copy(basis)
        cycle = [(min_i, min_j)]
        curr_i = min_i
        curr_j = min_j
        while True:
            dead_end = True
            for i, j in available_ways:
                if i == curr_i and j != curr_j and ((i, j) not in cycle or (i, j) == (min_i, min_j)):
                    if len(cycle) > 1 and cycle[-2][0] != i:
                        dead_end = False
                        curr_j = j
                        break
                    elif len(cycle) == 1:
                        dead_end = False
                        curr_j = j
                        break
                elif i != curr_i and j == curr_j and ((i, j) not in cycle or (i, j) == (min_i, min_j)):
                    if len(cycle) > 1 and cycle[-2][1] != j:
                        dead_end = False
                        curr_i = i
                        break
                    elif len(cycle) == 1:
                        dead_end = False
                        curr_i = i
                        break
            if not dead_end:
                cycle.append((curr_i, curr_j))
            elif cycle[0] != cycle[-1]:
                del available_ways[available_ways.index((curr_i, curr_j))]
                curr_i = min_i
                curr_j = min_j
                cycle = [(min_i, min_j)]
            else:
                break
        lam = math.inf
        for index, point in enumerate(cycle[:-1]):
            i, j = point
            if index % 2 and X[i][j] < lam:
                lam = X[i][j]
        deleted_from_basis = 0
        for index, point in enumerate(cycle[:-1]):
            i, j = point
            if index % 2:
                X[i][j] -= lam
            else:
                X[i][j] += lam
            if X[i][j] == 0 and deleted_from_basis == 0:
                del basis[basis.index((i, j))]
                deleted_from_basis += 1

    print('Таблица с исходными данными:')
    print_table(False)
    min_price_method()
    print('Начальный опорный план, полученный методом минимальной стоимости:')
    print_table()
    print(f'Стоимость перевозок по этому плану: {count_function()} единиц')
    X = np.zeros_like(C)  # Очитка массива X
    basis.clear()  # Очитка базиса
    northwest_corner_method()
    print('\033[101m' + 'Начальный опорный план, полученный методом северо-западного угла:' + '\033[0m')
    print_table(False)
    print(f'Стоимость перевозок по этому плану: {count_function()} единиц')
    count_potentials(True)
    print_table()
    count_delta(True)
    while np.min(delta) < 0 and num_iteration < 20:
        print('Остались отрицательные оценки, произведём перерасчёт плана...')
        num_iteration += 1
        print('\033[101m' + f'Итерация № {num_iteration}:' + '\033[0m')
        recalculate_optimal_plan()
        print_table()
        print(f'Стоимость перевозок по этому плану: {count_function()} единиц')
        count_potentials(True)
        count_delta(True)
    if num_iteration == 20:
        print(f'За {num_iteration} итераций не удалось найти оптимальное решение(')
    else:
        print('\033[101m' + 'Оптимальное решение найдено!' + '\033[0m')


def input_data():
    '''Ручной ввод данных'''
    поставщики = np.array(list(map(int, input(
        "Введите запас груза у каждого поставщика через пробел: ").split())))
    потребители = np.array(list(map(int, input(
        "Введите потребность груза у каждого потребителя через пробел: ").split())))
    c = list()  # Коэффициенты Cij
    for i in range(len(поставщики)):
        c.append(np.array(list(map(int, input(f"Введите стоимости перевозки от {
                 i + 1}-ого поставщика к каждому потребителю через пробел: ").split()))))
    return поставщики, потребители, c


def input_data_from_file():
    '''Автоматический ввод данных из файла'''
    with open('transportation_problem.csv', encoding='utf-8') as file:
        rows = csv.reader(file)
        поставщики = np.array(list(map(int, next(rows))))
        потребители = np.array(list(map(int, next(rows))))
        c = list()  # Коэффициенты Cij
        for row in rows:
            c.append(np.array(list(map(int, row))))
    return поставщики, потребители, c


transport_task()
