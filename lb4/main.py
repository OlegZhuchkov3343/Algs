from array import array


def calculate_lps(pattern):
    print("Вычисление префикс функции")
    n = len(pattern)
    lps = array('i', [0 for i in range(n)])
    j = 0
    i = 1
    while i < n:
        print(f"{i}: Длина префикса {j}; Сравниваем pattern[{i}] = {pattern[i]} с pattern[{j}] = {pattern[j]}")
        if pattern[i] == pattern[j]:
            j += 1
            print(f"\tСовпало; записываем {j} в pattern[{i}]; увеличиваем i")
            lps[i] = j
            i += 1
        else:
            if j != 0:
                print(f"\tНесовпадение; ставим длину {lps[j - 1]} (lps[{j} - 1])")
                j = lps[j - 1]
            else:
                print(f"\tНесовпадение; обнуляем lps[{i}]; увеличиваем i")
                lps[i] = 0
                i += 1
    print("Массив lps:")
    print(', '.join(map(str, lps)))
    return lps


def kmp(pattern, text):
    lps = calculate_lps(pattern)
    n = len(pattern)
    appearances = []
    i, j = 0, 0
    print("Поиск pattern в text")
    while i < len(text):
        print(f"Позиция в тексте {i} - {text[i]}, длина шаблона {j} - {pattern[j]}")
        if text[i] == pattern[j]:
            print("\tСовпадение, увеличиваем счетчики")
            i += 1
            j += 1
            if j == n:
                print(f"\tПолное вхождение шаблона с индекса {i-j}, переходим на длину шаблона lps[{j-1}] - {lps[j-1]}")
                appearances.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                print(f"Несовпадение, переходим на длину шаблона lps[{j-1}] - {lps[j-1]}")
                j = lps[j - 1]
            else:
                print("Несовпадение, увеличиваем счетчик текста")
                i += 1
    return appearances if appearances else [-1]


pattern = input()
text = input()
print(calculate_lps(pattern))
print(",".join(map(str, kmp(pattern, text))))

if len(pattern) == len(text):
    if pattern == text:
        print("Строки совпадают")
        print(0)
    else:
        print("Длины совпадают; проверка, является ли первая строка циклическим сдвигом второй:")
        print("Найдем вхождения второй строки в удвоенную первую строку")
        print(",".join(map(str, kmp(text, pattern*2))))
