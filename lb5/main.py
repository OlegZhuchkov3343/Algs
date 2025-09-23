class Node:
    def __init__(self, alphabet, number):
        self.children = dict(zip(alphabet, [None] * len(alphabet)))
        self.go = dict(zip(alphabet, [None] * len(alphabet)))
        self.parent = None
        self.suffix_link = None
        self.terminal_link = None
        self.char = ''
        self.is_leaf = False
        self.leaf_pattern_number = []
        self.text = None
        self.number = number

    def count_children(self):
        count = 0
        for i in self.children:
            if self.children[i]:
                count += 1
        return count


class Trie:
    def __init__(self, alphabet=('A', 'C', 'G', 'T', 'N')):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.root = Node(self.alphabet, 0)
        self.root.parent = self.root
        self.root.suffix_link = self.root
        self.root.terminal_link = self.root
        self.patterns = []
        self.node_count = 1

    def get_all_nodes(self):
        found = set()
        found.add(0)
        nodes = [self.root]
        node_queue = [self.root]
        while node_queue:
            cur = node_queue[0]
            for c in cur.children:
                v = cur.children[c]
                if v:
                    if v.number not in found:
                        nodes.append(v)
                        found.add(v)
                        node_queue.append(v)
            node_queue.pop(0)
        info_text = []
        for v in sorted(nodes, key=lambda x: x.number):
            info_text.append(f"[{v.number}] текст - {self.get_text(v)};\tпотомки: {', '.join(f"{v.children[i].char}->{v.children[i].number}" for i in v.children if v.children[i])};\t"
                f"суффиксная ссылка - {self.get_suffix_link(v, False).number if self.get_suffix_link(v, False) else "None"};\t"
                f"терминальная ссылка - {self.get_terminal_link(v, False).number if self.get_terminal_link(v, False) else "None"}")
        info_text.append(f"Макс. количество дуг, исходящих из вершины: {max([v.count_children() for v in nodes])}")
        return info_text

    def get_node_char(self, v):
        return v.char if v != self.root else 'root'

    def get_text(self, v):
        if v == self.root:
            return 'root'
        if v.text:
            return v.text
        result = ''
        while v != self.root:
            result = v.char + result
            v = v.parent
        v.text = result
        return result

    def get_link(self, v, c, verbose=True):
        if v.go[c] is None:
            if v.children[c] is not None:
                v.go[c] = v.children[c]
                if verbose:
                    print(f"\tПерейдем по бору {v.number} {self.get_text(v)} -> {v.go[c].number} {self.get_text(v.go[c])}")
            elif v == self.root:
                v.go[c] = self.root
                if verbose:
                    print(f"\tПерейдем в root")
            else:
                v.go[c] = self.get_link(self.get_suffix_link(v, verbose), c, verbose)
                if verbose:
                    print(f"\tПерейдем по суффиксной ссылке {v.number} {self.get_text(v)} -> {v.go[c].number} {self.get_text(v.go[c])}")
        else:
            if verbose:
                print(f"\tПереход по автомату {v.number} {self.get_text(v)} -> {v.go[c].number} {self.get_text(v.go[c])}")
        return v.go[c]

    def get_suffix_link(self, v, verbose=True):
        if v.suffix_link is None:
            if v == self.root or v.parent == self.root:
                v.suffix_link = self.root
                if verbose:
                    print(f"\tСтроим суффиксную ссылку на root")
            else:
                if verbose:
                    print(f"\tИщем суффикс в боре")
                v.suffix_link = self.get_link(self.get_suffix_link(v.parent, verbose), v.char, verbose)
                if v.suffix_link != self.root:
                    if verbose:
                        print("\tСуффикс найден")
                else:
                    if verbose:
                        print("\tМаксимальный суффикс пустой")
                if verbose:
                    print(f"\tСтроим суффиксную ссылку {v.number} {self.get_text(v)} -> {v.suffix_link.number} {self.get_text(v.suffix_link)}")
        else:
            if verbose:
                print(f"\tПереходим по суффиксной ссылке {v.number} {self.get_text(v)} -> {v.suffix_link.number} {self.get_text(v.suffix_link)}")
        return v.suffix_link

    def get_terminal_link(self, v, verbose=True):
        if v.terminal_link is None:
            suffix_link = self.get_suffix_link(v, verbose)
            if suffix_link.is_leaf:
                v.terminal_link = suffix_link
            elif suffix_link == self.root:
                v.terminal_link = self.root
            else:
                v.terminal_link = self.get_terminal_link(suffix_link, verbose)
            if verbose:
                print(f"\tСтроим терминальную ссылку {v.number} {self.get_text(v)} -> {v.terminal_link.number} {self.get_text(v.terminal_link)}")
        else:
            if verbose:
                print(f"\tПереходим по терминальной ссылке {v.number} {self.get_text(v)} -> {v.terminal_link.number} {self.get_text(v.terminal_link)}")
        return v.terminal_link

    def add_string(self, s, pattern_number):
        print(f"Добавим строку {s} в бор")
        cur = self.root
        for c in s:
            print(f"Находимся в {cur.number} {self.get_text(cur)}")
            if cur.children[c] is None:
                print(f"\tДобавляем {c}")
                new = Node(self.alphabet, self.node_count)
                new.char = c
                new.parent = cur
                cur.children[c] = new
                self.node_count += 1
            else:
                print(f"\t{c} уже существует")
            cur = cur.children[c]
        print(f"Находимся в {cur.number} {self.get_text(cur)}")
        print(f"\t{c} - терминальный символ")
        cur.is_leaf = True
        cur.leaf_pattern_number.append(pattern_number)
        self.patterns.append(s)

    def process_text(self, text):
        result = []
        current = self.root

        for i in range(len(text)):
            c = text[i]
            print(f"Рассмотрим вершину {c} на позиции {i + 1} в тексте {text}")
            current = self.get_link(current, c)
            if current == self.root:
                print("\tПодстрока не встречается в тексте")
            else:
                print(f"\tПерешли в состояние {self.get_text(current)}")

            temp_node = current
            while temp_node != self.root:
                if temp_node.is_leaf:
                    for num in temp_node.leaf_pattern_number:
                        pattern_length = len(self.patterns[num])
                        start_pos = i - pattern_length + 1
                        result.append((start_pos, num))
                        print(f"\tВершина {temp_node.number} терминальная, обнаружено вхождение подстроки {patterns[num]}")
                term = self.get_terminal_link(temp_node)
                temp_node = term
                print(f"\tПереходим по терминальной ссылке {temp_node.number} {self.get_text(temp_node)} -> {term.number} {self.get_text(term)}")

        print(f"Количество вершин в автомате = {self.node_count}")
        return result

    def process_text_with_mask(self, pattern, text, wildcard):
        if all(c == wildcard for c in pattern):
            print("Шаблон состоит только из масок")
            return []

        print("Разобьем строку на подстроки без маскок")
        substrings = list()
        substring_positions = list()
        i = 0
        while i < len(pattern):
            if pattern[i] == wildcard:
                i += 1
                continue
            start = i
            while i < len(pattern) and pattern[i] != wildcard:
                i += 1
            substrings.append(pattern[start:i])
            substring_positions.append(start)
        print(f"Подстроки без масок: {", ".join(substrings)} на позициях: {", ".join(map(str, substring_positions))}")

        print("Добавим подстроки в бор")
        for i in range(len(substrings)):
            self.add_string(substrings[i], i)

        counter = [0] * len(text)
        current = self.root
        print("Подсчитаем вхождения подстрок")
        for i in range(len(text)):
            c = text[i]
            print(f"Рассмотрим вершину {c} на позиции {i + 1} в тексте {text}")
            current = self.get_link(current, c)
            if current == self.root:
                print("\tПодстрока не встречается в тексте")
            else:
                print(f"\tПерешли в состояние {self.get_text(current)}")
            temp_node = current
            while temp_node != self.root:
                term = self.get_terminal_link(temp_node)
                if temp_node.is_leaf:
                    for num in temp_node.leaf_pattern_number:
                        substring_position = substring_positions[num]
                        substring_length = len(substrings[num])
                        start_pos = i - substring_length - substring_position + 1
                        if start_pos < 0:
                            continue
                        if start_pos + len(pattern) <= len(text):
                            counter[start_pos] += 1
                    print(f"\tВершина {temp_node.char} терминальная, обнаружено вхождение подстроки {self.patterns[num]}")
                temp_node = term

        print(f"Найдем вхождения шаблона")
        print(f"Получившийся счетчик совпадений: {counter}")
        result = []
        for i, count in enumerate(counter):
            if count == len(substrings):
                result.append(i + 1)
                print(f"\tКоличество вхождений совпало для позиции {i + 1} с числом {count}")
        print(f"Количество вершин в автомате = {self.node_count}")
        return result


var = int(input("Выберите вариант\n\t1: Поиск набора образцов\n\t2: Поиск образца с джокером\n"))
if var == 1:
    print("1. Текст; 2. Число N шаблонов; 3. N строк с шаблонами")
    text = input().strip()
    n = int(input())
    patterns = [input() for _ in range(n)]

    ac = Trie()
    print("Создание бора и добавление строк")
    for i, pattern in enumerate(patterns):
        ac.add_string(pattern, i)

    print("Преобразуем бор")
    matches = ac.process_text(text)

    print("Вывод вхождений в текст")
    matches.sort()
    mask = set()
    for pos, pattern_num in matches:
        print(pos+1, pattern_num+1)
    for pos, pattern_num in matches:
        print(f"Шаблон {patterns[pattern_num]} встречается в тексте {text} на позиции {pos}")
        for i in range(pos, pos+len(patterns[pattern_num])):
            mask.add(i)
    cut = ""
    for i in range(len(text)):
        if i not in mask:
            cut += text[i]

    print("\n".join(ac.get_all_nodes()))
    print("Текст с вырезанными фрагментами:")
    print(cut)


elif var == 2:
    text = input()
    wildcard_pattern = input()
    wildcard = input()

    ac = Trie()
    matches = ac.process_text_with_mask(wildcard_pattern, text, wildcard)

    print(f"Вывод найденных вхождений шаблона")
    print(f"Шаблон {wildcard_pattern} встречается в тексте {text} на позициях {", ".join(map(str, matches))}.")
    print("\n".join(ac.get_all_nodes()))
    print("Текст с вырезанными фрагментами:")
    cut = ""
    mask = set()
    for pos in matches:
        for i in range(pos-1, pos-1+len(wildcard_pattern)):
            mask.add(i)
    for i in range(len(text)):
        if i not in mask:
            cut += text[i]
    print(cut)
