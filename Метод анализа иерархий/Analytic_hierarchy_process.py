import pandas as pd
import functools


NUM_CRITERIA = 5  # Количество критериев для сравнения
NUM_ALTERNATIVES = 5  # Количество альтернатив
СИ = 1.12  # среднее значение индекса случайной согласованности


def print_table(data):
    '''Функция для вывода таблицы'''
    print(pd.DataFrame(data).to_markdown())


def relative_value(data, MATRIX_SIZE=NUM_CRITERIA):
    '''Функция для определения относительной ценности элемента (геометрическое среднее)'''
    return round(functools.reduce(lambda a, b: a * b, data) ** (1 / MATRIX_SIZE), 3)


def compare_by_criteria(data, index):
    '''Функция для сравнения в пределах матрицы сравнения по критерию N'''
    V = []
    for i in range(NUM_ALTERNATIVES):
        val = relative_value(data[i])
        print(
            f'Строка №{i + 1}\nVk{index}{i+1} = ({" * ".join([str(i) for i in data[i]])}) ^ 1/{NUM_ALTERNATIVES} = {val}')
        V.append(val)
    print(
        f'Проведена нормализация полученных чисел. Для этого определен нормирующий коэффициент ∑VK{index}Y.')
    print(
        f'∑VК{index}Y  = VK{index}1 + VK{index}2 + VK{index}3 + VK{index}4 + VK{index}5 = {" + ".join([str(i) for i in V])} = {round(sum(V), 3)}.')
    print(
        f'Найдена важность приоритетов W3К{index}Y, для этого каждое из чисел VK{index}Y разделено на ∑VK{index}Y.')
    Y = []
    for i in range(NUM_ALTERNATIVES):
        Y.append(round(V[i] / sum(V), 3))
        print(
            f'Строка №{i + 1}\nW3K{index}{i + 1} = {V[i]} / ∑Vi = {V[i]} / {sum(V)} = {Y[i]};')
    print('В результате получаем вектор приоритетов:')
    print(f'W3K{index}Y = ({"; ".join([f"Y3{index}{i + 1} = {Y[i]}" for i in range(len(Y))])}), '
          + f'где индекс 3 означает, что вектор приоритетов относится к третьему уровню иерархии критерия K{index}.')
    return Y


def check_matrix_consistency(data, priority_vector, index):
    '''Функция для определения согласованности матрицы'''
    print(
        f'Определены индекс согласованности и отношение согласованности для матрицы К{index}')
    print('Определяется сумма каждого столбца матрицы суждений.')
    counter = 1
    S = []
    for i in zip(*data):
        S.append(sum(i))
        print(
            f'S{counter}K{index} = {" + ".join([str(i) for i in list(i)])} = {sum(i)}')
        counter += 1
    print('Затем полученный результат умножен на компоненту нормализованного вектора приоритетов.')
    P = []
    for i in range(len(S)):
        P.append(round(S[i] * priority_vector[i], 3))
        print(f'P{i + 1}K{index} = S{i + 1} * W3K{index}{i + 1} = {P[i]}')
    print('Найдена пропорциональность предпочтений.')
    print(f'λmaxK{index} = Р1K{index} + Р2K{index} + Р3K{index} + Р4K{index} + Р5K{index} = {round(sum(P), 3)}')
    print('Отклонение от согласованности выражается индексом согласованности.')
    ИС = round((round(sum(P), 3) - 5) / (5 - 1), 3)
    print(
        f'ИСk{index} = (λmaxK{index} - n)/(n - 1) = ({round(sum(P), 3)}-5)/(5-1) = {ИС}.')
    print('Найдено отношение согласованности ОС.')
    print(f'ОСk{index} = ИС/СИ = {round(ИС / СИ, 3)}.')


def synthesis_of_alternatives(Y, full_Y):
    print(f'Векторы приоритетов и отношения согласованности определяются для всех матриц суждений, начиная со второго уровня.\n' +
          f'Для определения приоритетов альтернатив локальные приоритеты умножены на приоритет соответствующего критерия' +
          f'на высшем уровне и найдены суммы по каждому элементу в соответствии с критериями, на которые воздействует этот элемент.')
    print(
        f'W2i = ({"; ".join([f"Y2{i + 1} = {Y[i]}" for i in range(len(Y))])});')
    for i in range(len(full_Y)):
        print(
            f'W3K{i + 1}Y = ({"; ".join([f"Y3{i + 1}{j + 1} = {full_Y[i][j]}" for j in range(len(full_Y[i]))])});')
    print('Приоритеты альтернатив получены следующим образом:')
    winners = []
    counter = 0
    for v in zip(*full_Y):
        curr_str = str()
        w = 0
        counter += 1
        curr_str += f'W{counter} = '
        for i in range(len(v)):
            curr_str += f'W2{i + 1} * W3K{i + 1}{counter} + '
            w += (v[i] * Y[i])
        w = round(w, 3)
        print(curr_str.rstrip(' + ') + ' = ' + str(w))
        winners.append(w)
    print('Таким образом, приоритеты альтернатив равны:')
    for i in range (NUM_ALTERNATIVES):
        print(f'альтернатива А{i+1} - W{i + 1} приоритет равен {winners[i]}')
    return winners


with open('Analytic_hierarchy_process.csv', encoding='utf-8') as file:
    title = file.readline().rstrip()  # Текущая матрица, которая будет считана
    print(title.center(201))
    criteria_paired_comparison_matrix = [[float(i) for i in file.readline().rstrip(
    ).split(',')] for _ in range(NUM_CRITERIA)]  # Матрица парного сравнения критериев
    print_table(criteria_paired_comparison_matrix)
    print('Для определения относительной ценности каждого элемента необходимо найти геометрическое' +
          ' среднее и с этой целью перемножить n элементов каждой строки и из полученного' +
          ' результата извлечь корни n-й степени (размерность матрицы n=5).')
    V = []
    for i in range(NUM_CRITERIA):
        val = relative_value(criteria_paired_comparison_matrix[i])
        print(
            f'Строка №{i + 1}\nV{i + 1} = ({" * ".join([str(i) for i in criteria_paired_comparison_matrix[i]])}) ^ 1/{NUM_CRITERIA} = {val}')
        V.append(val)
    print('Проведена нормализация полученных чисел. Для этого определен нормирующий коэффициент ∑Vi.')
    print(
        f'∑Vi = V1 + V2 + V3 + V4 + V5 = {" + ".join([str(i) for i in V])} = {round(sum(V),3)}')
    print('Найдена важность приоритетов W2i, для этого каждое из чисел Vi разделено на ∑Vi.')
    Y = []  # Вектор приоритетов W2i
    for i in range(NUM_CRITERIA):
        Y.append(round(V[i] / sum(V), 3))
        print(
            f'Строка №{i + 1}\nW2{i + 1} = {V[i]} / ∑Vi = {Y[i]} = Y{2}{i + 1}')
    print(f'В результате получен вектор приоритетов:\nW2i = ({"; ".join([f"Y2{i + 1} = {Y[i]}" for i in range(len(Y))])}), '
          + 'где индекс 2 означает, что вектор приоритетов относится ко второму уровню иерархии.')
    big_data, full_Y = [], []
    for i in range(NUM_CRITERIA):
        title = file.readline().rstrip()
        print(title.center(201))
        data = [[float(i) for i in file.readline().rstrip().split(',')] for _ in range(
            NUM_ALTERNATIVES)]  # Матрица сравнения по i + 1-ому критерию
        big_data.append(data)
        print_table(data)
        Yi = compare_by_criteria(data, i + 1)
        full_Y.append(Yi)
    print('Определены индекс согласованности и отношение согласованности для матрицы «Выбор лучшего технического вуза»')
    print('Определена сумма каждого столбца матрицы суждений.')
    counter = 1
    S = []
    for i in zip(*criteria_paired_comparison_matrix):
        S.append(sum(i))
        print(
            f'S{counter} = {" + ".join([str(i) for i in list(i)])} = {sum(i)}')
        counter += 1
    print(f'Полученный результат умножен на компоненту нормализованного вектора приоритетов, ' +
          f'т.е. сумму суждений первого столбца на первую компоненту, сумму суждений второго столбца - на вторую и т.д.')
    P = []
    for i in range(len(S)):
        P.append(round(S[i] * Y[i], 3))
        print(f'P{i + 1} = S{i + 1} * W2{i + 1} = {P[i]}')
    print(f'Сумма чисел Рj отражает пропорциональность предпочтений, ' +
          f'чем ближе эта величина к n (числу объектов и видов действия в матрице парных сравнений), тем более согласованны суждения.')
    print(f'λmax = Р1 + Р2 + Р3 + Р4 + Р5 = {round(sum(P), 3)}')
    print('Отклонение от согласованности выражается индексом согласованности.')
    ИС = round((round(sum(P), 3) - 5) / (5 - 1), 3)
    print(f'ИС = (λmax - n)/(n - 1) = ({round(sum(P), 3)}-5)/(5-1) = {ИС}.')
    print('Отношение индекса согласованности ИС к среднему значению случайного индекса согласованности СИ называется отношением согласованности ОС.')
    print(f'ОС = ИС/СИ = {round(ИС / СИ, 3)}.')
    for i in range(NUM_CRITERIA):
        check_matrix_consistency(big_data[i], full_Y[i], i + 1)
    synthesis_of_alternatives(Y, full_Y)
