import re
import math
import sympy
import numpy as np


NUM_CRITERIA = 4  # Количество ограничений в математической моделе
PRECISION = 8  # Количество знаков после запятой при округлении
SEP = 25  # Разделитель для вывода таблицы


def print_table(system, coef_basis, coef_not_basis, basis_values, not_basis_values):
    print(' '.ljust(SEP), end='')
    print('Cj'.ljust(SEP), end='')
    for i in range(len(coef_not_basis) + 1):
        if i == len(coef_not_basis):
            print(' '.ljust(SEP))
        else:
            print(str(coef_not_basis[i]).ljust(SEP), end='')
    print('Cv'.ljust(SEP), end='')
    for i in range(len(not_basis_values) + 1):
        if i == 0:
            print(''.ljust(SEP), end='')
        else:
            print(str(not_basis_values[i-1]).ljust(SEP), end='')
    print('A0'.ljust(SEP))
    system.insert(0, coef_basis + [' '])
    system.insert(1, basis_values + ['f'])
    for column in range(len(system[0])):
        for row in range(len(system)):
            print(str(system[row][column]).ljust(SEP), end='')
        print()
    del system[0]
    del system[0]


def get_coefficients(data):
    '''Функция для получения списка коэффициентов из системы ограничений'''
    criteria_coefficients, boundaries = list(), list()
    for exp in data:
        if '<=' in exp:
            parse = exp.split('<=')
        elif '>=' in exp:
            parse = exp.split('>=')
        elif '<' in exp:
            parse = exp.split('<')
        elif '>' in exp:
            parse = exp.split('>')
        elif '=' in exp:
            parse = exp.split('=')
        parse = list(map(str.strip, parse))
        boundaries.append(float(parse[1]))
        criteria_coefficients.append(list(map(float, [i.group(1) for i in re.finditer(
            r'(\d+(\.\d+)?) {0,}[*]? {0,}\w', parse[0])])))
    return criteria_coefficients, boundaries


def count_scalar_product(vec1, vec2):
    '''Функция для рассчёта скалярного произведения двух векторов'''
    res = 0
    for i in range(len(vec1)):
        res += (vec1[i] * vec2[i])
    return res


def create_simplex_table(system, coef_basis, coef_not_basis, basis_values, not_basis_values):
    F_str = [0] * len(not_basis_values)
    for i in range(len(not_basis_values)):
        F_str[i] = count_scalar_product(
            coef_basis, system[i]) - coef_not_basis[i]
    Q = count_scalar_product(coef_basis, system[-1])
    for i in range(len(F_str)):
        system[i].append(F_str[i])
    system[-1].append(Q)
    return F_str, Q


def simplex_iteration(system, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q):
    index_column = F_str.index(min(F_str))
    mini = 1e10
    for i in range(len(system[-1]) - 1):
        tmp = system[-1][i] / system[index_column][i]
        if tmp < mini:
            mini = tmp
            index_row = i
    key_element = system[index_column][index_row]
    basis_values.insert(index_row, not_basis_values[index_column])
    not_basis_values.insert(index_column, basis_values.pop(index_row + 1))
    del not_basis_values[index_row]
    coef_basis[index_row], coef_not_basis[index_column] = coef_not_basis[index_column], coef_basis[index_row]
    new_key_element = round(1 / key_element, PRECISION)
    data = [[0] * (len(coef_basis) + 1)
            for _ in range(len(coef_not_basis) + 1)]
    for i in range(len(system[index_column])):
        data[index_column][i] = - \
            round(system[index_column][i] / key_element, PRECISION)
    for i in range(len(system)):
        data[i][index_row] = round(
            system[i][index_row] / key_element, PRECISION)
    data[index_column][index_row] = new_key_element
    for row in range(len(data[0])):
        for column in range(len(data)):
            if data[column][row] == 0:
                data[column][row] = round(((system[column][row] * key_element) - (
                    system[index_column][row] * system[column][index_row])) / key_element, PRECISION)
    F_str = [data[i][-1] for i in range(len(data) - 1)]
    Q = data[-1][-1]
    return data, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q


def check_inequality(inequality, variables, индексы_базисных_переменных_оптимального_плана):
    inequality = inequality.replace('*', '')
    for i in range(len(variables)):
        if variables[i] not in inequality:
            continue
        inequality = inequality.replace(
            variables[i], '*' + str(индексы_базисных_переменных_оптимального_плана[i]))
    # Символьное вычисление неравенства
    inequality += '- 0.1'
    result = str(sympy.sympify(inequality))
    # Возвращение True, если неравенство выполняется, иначе False
    return eval(result)


def dual_task():
    коэффициенты_целевой_функции = np.array(target_coefficients)
    свободные_члены_неравенств = np.array(boundaries)
    матрица_ограничений, _ = get_coefficients(criteria_function)
    транспонированная_матрица_ограничений = np.transpose(матрица_ограничений)
    индексы_базисных_переменных_оптимального_плана = np.array(system[-1][:-1])
    y = np.array([])
    D = list()
    for i in range(len(basis_values)):
        index = int(basis_values[i][1:]) - 1
        if index < len(транспонированная_матрица_ограничений):
            D.append(транспонированная_матрица_ограничений[index])
        else:
            D.append(
                np.array([1 if i == j else 0 for j in range(NUM_CRITERIA)]))
    D_inversed = np.linalg.inv(np.transpose(D))

    def first_duality_theorem():
        y = np.dot(np.array(coef_basis), D_inversed)
        G_min = np.dot(свободные_члены_неравенств, y)
        print(f"Gmin is {G_min} by first_duality_theorem")
        assert abs(G_min - Q) < 0.00001

    def second_duality_theorem():
        nonlocal y
        zeros = list()
        for i in range(NUM_CRITERIA):
            if check_inequality(
                    criteria_function[i], basis_values, индексы_базисных_переменных_оптимального_плана):
                zeros.append(i)
        система_уравнений = транспонированная_матрица_ограничений.copy()
        for i in range(len(zeros)):
            система_уравнений = np.delete(
                система_уравнений, zeros[i], 1)
            for j in range(i + 1, len(zeros)):
                zeros[j] -= 1
        y = np.linalg.solve(система_уравнений, коэффициенты_целевой_функции)
        for i in range(len(zeros)):
            y = np.insert(y, zeros[i], 0)
            for j in range(i + 1, len(zeros)):
                zeros[j] += 1
        G_min = np.dot(свободные_члены_неравенств, y)
        print(f"Gmin is {G_min} by second_duality_theorem")
        assert abs(G_min - Q) < 0.00001

    def third_duality_theorem():
        нижняя_граница = list()
        верхняя_граница = list()
        b = list()
        for i in range(len(D_inversed) - 1, -1, -1):
            pozitive = list()
            negative = list()
            bH = - math.inf
            bB = math.inf
            for j in range(len(D_inversed)):
                if D_inversed[j][i] > 0:
                    pozitive.append(
                        (свободные_члены_неравенств[j], D_inversed[j][i]))
                elif D_inversed[j][i] < 0:
                    negative.append(
                        (свободные_члены_неравенств[j], D_inversed[j][i]))
            if len(pozitive) > 1:
                elem = min(pozitive, key=lambda x: abs(
                    pozitive[0][0] / pozitive[0][1]))
                нижняя_граница.append(elem[0] / elem[1])
            elif len(pozitive) == 1:
                нижняя_граница.append(pozitive[0][0] / pozitive[0][1])
            else:
                нижняя_граница.append(bH)

            if len(negative) > 1:
                elem = max(negative, key=lambda x: abs(
                    negative[0][0] / negative[0][1]))
                верхняя_граница.append(abs(elem[0] / elem[1]))
            elif len(negative) == 1:
                верхняя_граница.append(negative[0][0] / negative[0][1])
            else:
                верхняя_граница.append(bB)

            b.append(свободные_члены_неравенств[i])
            print(f'Ресурс №{len(D_inversed)-i}')
            print(
                f'b{len(D_inversed)-i} ∈ ({нижняя_граница[-1]}; {верхняя_граница[-1]})')
            print(f'{len(D_inversed)-i}-й ресурс изменяется в интервале: ', end='')
            if нижняя_граница[-1] == - math.inf:
                print(f'({нижняя_граница[-1]}; ', end='')
            else:
                print(f'({b[-1] - нижняя_граница[-1]}; ', end='')
            if верхняя_граница[-1] == math.inf:
                print(f'{верхняя_граница[-1]})')
            else:
                print(f'{b[-1] + верхняя_граница[-1]})')
        total = 0
        for i in range(len(y)):
            if y[i] != 0:
                total += y[i] * верхняя_граница[i]
                print(f'∆Gmax{i + 1} = y{i+1} * bB{i +
                      1} = {y[i] * верхняя_граница[i]}')
        print(f'Совместное влияние изменений этих ресурсов приводит к изменению максимальной стоимости продукции 𝐺𝑚𝑎𝑥 на величину: {total}')
        print(f'Следовательно, оптимальное значение целевой функции при максимальном изменении ресурсов: {Q+total}')

    first_duality_theorem()
    second_duality_theorem()
    third_duality_theorem()


with open(r'C:\projects\MIREA\Теория принятия решений\Симплексный метод\simplex_method.csv', encoding='utf-8') as file:
    target_function = file.readline().rstrip()  # Целевая функция
    target_coefficients = list(map(float, [i.group(1) for i in re.finditer(
        # Список коэффициентов целевой функции
        r'(\d+(\.\d+)?) {0,}[*]? {0,}\w', target_function)]))
    criteria_function = [file.readline().rstrip() for _ in range(NUM_CRITERIA)]
    criteria_coefficients, boundaries = get_coefficients(criteria_function)
    print('Переходим к задаче линейного программирования:',
          target_function, sep='\n')
    for i in criteria_function:
        print("{ " + i)
    system = list(map(list, list(zip(*criteria_coefficients))))
    system.append(boundaries.copy())
    # Вектор коэффициентов целевой функции при базисных переменных
    coef_basis = [0] * NUM_CRITERIA
    # Коэффициенты целевой функции, соответствующие небазисным переменным
    coef_not_basis = target_coefficients.copy()
    not_basis_values = re.findall(r'[A-Za-z]\d{1,}', target_function)
    basis_values = [f'{not_basis_values[-1][0]}{i}' for i in range(
        int(not_basis_values[-1][1]) + 1, NUM_CRITERIA + int(not_basis_values[-1][1]) + 1)]
    F_str, Q = create_simplex_table(
        system, coef_basis, coef_not_basis, basis_values, not_basis_values)
    num_iteration = 0
    while num_iteration < 50 and min(F_str) < 0:
        print(
            ('\x1b[6;30;42m' + f"Итерация №{num_iteration}" + '\x1b[0m').center(201))
        system, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q = simplex_iteration(
            system, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q)
        print_table(system, coef_basis, coef_not_basis,
                    basis_values, not_basis_values)
        num_iteration += 1
    if num_iteration != 50:
        print(f'Решение найдено! Общая прибыль составила {
              round(Q, 3)} денежных единиц')
    else:
        print('Поставленная задача решения не имеет')
        exit(0)
    dual_task()
