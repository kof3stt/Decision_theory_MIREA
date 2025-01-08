import csv
import pandas as pd


def print_table(data):
    '''Функция для вывода таблицы'''
    print(pd.DataFrame(data).to_markdown())


def compare_alternatives(a, b):
    '''Функция для сравнения  альтернатив по отношению Парето-доминирования'''
    counter = 0
    for key in a:
        if '+' in key:
            counter += (float(a[key]) > float(b[key]))
        elif '-' in key:
            counter += (float(a[key]) < float(b[key]))
    return 1 if counter == len(a) - 1 else -1 if counter == 0 else 0


def create_Pareto_set(data):
    '''Функция для создания оптимального множества Парето по входящему множеству альтернатив'''
    losers, winners = [], []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            n = compare_alternatives(data[i], data[j])
            if n == 1:
                losers.append(data[j])
            elif n == -1:
                losers.append(data[i])
    for i in range(len(data)):
        if data[i] not in losers:
            winners.append(data[i])
    return winners


def branches_and_boundaries(data, branches):
    '''Метод указания верхних и нижних границ критериев'''
    winners = []
    for i in data:
        flag = False
        for j in branches:
            key, value = list(j.items())[0]
            if key.count('-'):
                if float(i[key]) > value:
                    flag = True
            else:
                if float(i[key]) < value:
                    flag = True
        if not flag:
            winners.append(i)
    return create_Pareto_set(winners)


def suboptimization(data, branches, main_criteria):
    '''Метод субоптимизации'''
    data = branches_and_boundaries(data, branches)
    maxi = max(data, key=lambda i: i[main_criteria])
    return list(filter(lambda x: x[main_criteria] == maxi[main_criteria], data)) 


def lexical_optimization(data, priority):
    '''Лексикографический метод'''
    return [max(data, key = lambda item: tuple(item[key] for key in priority))]


with open('Pareto_method.csv', encoding='utf-8') as file:
    data = [d for d in csv.DictReader(file)]

    print("Исходная таблица с альтернативами и критериями:".center(201))
    print_table(data)

    print("Оптимальное-множество Парето:".center(201))
    print_table(create_Pareto_set(data))

    print("Установка верхних и нижних границ:".center(201))
    branches = [{"Проходной балл (+)": 270}, {"Рейтинг университета (+)": 840},
                {"Расстояние до общежития (-)": 14}]
    print_table(branches_and_boundaries(data, branches))

    print("Субоптимизация:".center(201))
    branches = [{"Проходной балл (+)": 290}, {"Расстояние до общежития (-)": 14}]
    main_criteria = "Рейтинг университета (+)"
    print_table(suboptimization(data, branches, main_criteria))

    print("Лексикографическая оптимизация:".center(201))
    priority = ("Рейтинг университета (+)", "Проходной балл (+)", "Стоимость обучения (+)",
                "Кол-во бюджетных мест (-)", "Расстояние до общежития (-)",
                "Размер стипендии (+)")
    print_table(lexical_optimization(data, priority))
