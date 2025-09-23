def print_D(str1, str2, D, path, curse):
    path = set(path)
    print("# - проклятый символ, * - пройденный путь")
    print("\t".join([" ", " "] + list(str2)))
    print("\n".join("\t".join([("#" + str1[row - 1] if row - 1 in curse else str1[row - 1]) if row != 0 else " "] +
                              [f"*{D[row][col]}" if (row, col) in path else str(D[row][col]) for col in
                               range(len(D[0]))])
                    for row in range(len(D))))


def wagner_fischer_table(str1, str2, costs, curse):
    print("Заполняем матрицу по алгоритму Вагнера Фишера")
    replace_cost, insert_cost, delete_cost = costs
    n, m = len(str1) + 1, len(str2) + 1
    D = [[0 for j in range(m)] for i in range(n)]
    for x in range(1, m):
        D[0][x] = D[0][x - 1] + insert_cost
    print("Первая строка:", ' '.join(map(str, D[0])))
    for y in range(1, n):
        cursed, special = False, False
        print(f"Заполняем {y}, {0}; ", end='')
        if y-1 in curse:
            cursed = True
            print("проклятый символ; ", end='')
            if str1[y-1].upper() == "Z":
                print("но можно заменить; ", end='')
                special = True
        if D[y-1][0] == "inf" or cursed:
            D[y][0] = "inf"
            print(f"Значение inf - невозможная позиция")
        else:
            D[y][0] = D[y-1][0] + delete_cost
            print(f"Значение {D[y-1][0]} + {delete_cost}: {D[y][0]}")
        for x in range(1, m):
            print(f"Заполняем {y}, {x}:")
            options = set()
            if D[y-1][x] != "inf" and not cursed:
                print(f"\tВозможно удаление - {D[y-1][x]} + {delete_cost}: {D[y-1][x] + delete_cost}")
                options.add(D[y-1][x] + delete_cost)
            if D[y][x-1] != "inf":
                print(f"\tВозможна вставка - {D[y][x-1]} + {insert_cost}: {D[y][x-1] + insert_cost}")
                options.add(D[y][x-1] + insert_cost)
            if D[y-1][x-1] != "inf" and (not cursed or special):
                print(f"\tВозможна замена - {D[y-1][x-1]} + {replace_cost}: {D[y-1][x-1] + replace_cost}")
                options.add(D[y-1][x-1] + replace_cost)
            if str1[y-1] != str2[x-1]:
                if options:
                    print(f"\tВыбранное значение: {min(options)}")
                    D[y][x] = min(options)
                else:
                    print("\tЗначение inf - невозможная позиция")
                    D[y][x] = "inf"
            else:
                D[y][x] = D[y-1][x-1]
                print(f"\tОдинаковый символ: {D[y][x]}")
    return D


def redact(str1, str2, D):
    print("Ищем редакционное предписание")
    if D[-1][-1] == "inf":
        return "Conversion impossible", []
    n, m = len(str1) + 1, len(str2) + 1

    instruction = str()

    y, x = n - 1, m - 1
    coords = [(y, x)]

    while y > 0 or x > 0:
        print(f"Точка {x}, {y}")
        if y == 0:
            print("\tв верхней строке, идем влево, добавляем I в начало")
            instruction = "I" + instruction
            x -= 1
            coords.append((y, x))
            continue
        if x == 0:
            print("\tв левом столбце, идем вверх, добавляем D в начало")
            instruction = "D" + instruction
            y -= 1
            coords.append((y, x))
            continue
        delete, insert, replace = D[y - 1][x], D[y][x - 1], D[y - 1][x - 1]
        options = [option for option in [delete,insert,replace] if option != "inf"]
        print(f"\tВозможные операции: {', '.join(map(str,options))}")
        min_operation = min(options)
        if min_operation == replace:
            if str1[y - 1] != str2[x - 1]:
                print(f"\t{min_operation} - Замена, добавляем R в начало")
                print("\tИдем по диагонали")
                instruction = 'R' + instruction
            else:
                print(f"\t{min_operation} - Пропуск, добавляем M в начало")
                print("\tИдем по диагонали")
                instruction = 'M' + instruction
            y -= 1
            x -= 1
        elif min_operation == insert:
            print(f"\t{min_operation} - Вставка, добавляем I в начало")
            print("\tИдем влево")
            instruction = 'I' + instruction
            x -= 1
        elif min_operation == delete:
            print(f"\t{min_operation} - Удаление, добавляем D в начало")
            print("\tИдем вверх")
            instruction = 'D' + instruction
            y -= 1
        coords.append((y, x))

    return instruction, coords


print("1: first string\n2: second string\n3: operation costs separated by space (replace, insert, delete)\n4: cursed indexes in first string separated by space")
A = input()
B = input()
costs = input()
curse = input()
costs = list(map(int, costs.split()))
if curse:
    curse = set(map(int, curse.split()))
D = wagner_fischer_table(A, B, costs, curse)
print()
distance = D[-1][-1]
instruction, path = redact(A, B, D)
print_D(A, B, D, path, curse)
print(distance, instruction)
