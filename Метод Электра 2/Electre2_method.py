import csv
import math
from graphviz import Digraph


def print_matrix(c = 1):
    '''Функция для вывода матрицы предпочтений с порогом'''
    print('-' * (10 * (len(matrix) + 1) + 4))
    print(11 * ' ', end = '')
    for i in range(1, len(matrix) + 1):
        print(f'{i}'.ljust(11), end = '')
    print('\n' + '-' * (10 * (len(matrix) + 1) + 4))
    for i in range(len(matrix)):
        print(str(i + 1).ljust(10), end='|')
        for j in range(len(matrix)):
            if matrix[i][j] < c:
                matrix[i][j] = 0
            print(str(matrix[i][j]).ljust(10), end=' ')
        print()
    print('-' * (10 * (len(matrix) + 1) + 4))


def compare_alternatives(i, j, alt_i, alt_j, criteria):
    '''Функция для сравнения альтернатив по кодам'''
    P, N = 0, 0
    for i in range(len(criteria)):
        counter_i, counter_j = 0, 0
        code = criteria[i]['Код'].split(';')
        for border in criteria[i]['Шкала'].split(';')[1:]:
            if alt_i[i] < float(border):
                counter_i += 1
            if alt_j[i] < float(border):
                counter_j += 1
        alt_i[i] = float(code[counter_i])
        alt_j[i] = float(code[counter_j])
        if criteria[i]['Стремление'] == '-':
            alt_i[i] *= (-1)
            alt_j[i] *= (-1)
        if alt_i[i] > alt_j[i]:
            P += int(criteria[i]['Вес критерия'])
        elif alt_i[i] < alt_j[i]:
            N += int(criteria[i]['Вес критерия'])
    return P, N


def generate_matrix():
    '''Функция для генерации матрицы предпочтений'''

    def generate_text(i, j, alt_i, alt_j):
        '''Функция для генерации вспомогательного текста при формировании матрицы'''
        print(f'Рассмотрим альтернативы {i} и {j} (i = {i}, j = {j}):')
        P_STR, N_STR = f'P{i}{j} =', f'N{i}{j} ='
        for k in range(len(criteria)):
            if alt_i[k] > alt_j[k]:
                P_STR += (' ' + criteria[k]['Вес критерия'] + ' +')
                N_STR += (' ' + str(0) + ' +')
            elif alt_i[k] < alt_j[k]:
                N_STR += (' ' + criteria[k]['Вес критерия'] + ' +')
                P_STR += (' ' + str(0) + ' +')
            else:
                N_STR += (' ' + str(0) + ' +')
                P_STR += (' ' + str(0) + ' +')
        P_STR = P_STR.rstrip(' +') + f' = {P}'
        N_STR = N_STR.rstrip(' +') + f' = {N}'
        print(P_STR+';', N_STR+';', sep='\n')
        print(generate_D_STR(i, j, P, N))
        print(f'P{j}{i}{N_STR[3:]};', f'N{j}{i}{P_STR[3:]};', sep='\n')
        print(generate_D_STR(j, i, N, P))

    def get_D(P, N):
        '''Функция для расчёта D-стремления'''
        if N == 0 and P == 0:
            return 1
        elif N == 0 and P != 0:
            return math.inf
        value = P/N
        if math.floor(value) == math.ceil(value):
            value = int(value)
        else:
            value = round(value, 2)
        return value

    def generate_D_STR(i, j, P, N):
        '''Функция для генерации D-стремления'''
        value = get_D(P, N)
        if value <= 1:
            return f'D{i}{j} = P{i}{j} / N{i}{j} = {P}/{N} = {value} <= 1 - отбрасываем.'
        else:
            if value == math.inf:
                value = '\u221e'
            return f'D{i}{j} = P{i}{j} / N{i}{j} = {P}/{N} = {value} > 1 - принимаем.'

    for i in range(1, len(data) + 1):
        for j in range(i + 1, len(data) + 1):
            alt_i, alt_j = data[i-1].copy(), data[j-1].copy()
            P, N = compare_alternatives(i, j, alt_i, alt_j, criteria)
            generate_text(i, j, alt_i, alt_j)
            D = get_D(P, N)
            if get_D(P, N) > 1:
                matrix[i - 1][j - 1] = D
            D = get_D(N, P)
            if D > 1:
                matrix[j - 1][i - 1] = D


def draw_graph(c = 1):
    '''Функция для рисования хаотичного графа с порогом'''
    dot = Digraph(f'Хаотичный Граф с порогом = {c}')
    for i in range(len(matrix)):
        dot.node(str(i + 1))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] >= c:
                dot.edge(str(i + 1), str(j + 1))
    dot.render(view=True)


def smart_draw_graph(levels, c = 1):
    '''Функция для рисования графа по уровням с порогом'''
    dot1 = Digraph(f"Граф с порогом = {c}")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] >= c:
                dot1.edge(str(i + 1), str(j + 1))
    for i in range(len(levels)):
        sub = Digraph(name = 'Подграф' + str(i))
        sub.attr(rank = 'same')
        sub.node(f'{i + 1}-ый уровень')
        for j in levels[i]:
            sub.node(f'{j + 1}')
        dot1.subgraph(sub)
    dot1.render(view = True)


def get_levels(c = 1):
    '''Вспомогательная функция для определения уровня вершин'''
    levels = []  # массив всех вершин
    visited = []  # массив посещённых вершин
    while len(visited) < len(matrix):
        level = []
        for i in range(len(matrix)):
            if i in visited:
                continue
            flag = True
            for j in range(len(matrix)):
                if matrix[j][i] >= c:
                    flag = any(j in lev for lev in levels)
                if not flag:
                    break
            if flag:
                level.append(i)
                visited.append(i)
        levels.append(level)
        print(f'{len(levels)}-ый уровень: ' +
              ', '.join(map(lambda x: str(x + 1), level)))
    return levels


with open('Electre2_method.csv', encoding = 'utf-8') as file, \
        open('codes.csv', encoding = 'utf-8') as criteria_file:
    criteria = [i for i in csv.DictReader(criteria_file)]  # Информация о критериях
    data = list(map(lambda x: [float(i) for i in x], [
                i[1:] for i in csv.reader(file)][1:]))  # Значения критериев для рассматриваемых альтернатив
    matrix = [[0] * len(data) for _ in range(len(data))]  # Матрица предпочтений
    generate_matrix()
    print("Матрица предпочтений:".center(201))
    print_matrix()
    draw_graph()
    arg = 1.76
    smart_draw_graph(get_levels(c = arg), c = arg)
