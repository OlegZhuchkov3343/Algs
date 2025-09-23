from modules.interface import TkWindow
from math import ceil


class Square:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.right = x + size - 1
        self.bottom = y + size - 1
        self.size = size

    def __str__(self):
        return f"({self.x},{self.y}): {self.size}"


class Solution:
    def __init__(self, count, squares, scale, grid_size):
        self.count = count
        self.squares = squares
        self.operation_count = 0
        self.log = list()
        self.scale = scale
        self.grid_size = grid_size
        self.solution_index = -1

    def print(self, scale):
        print(self.count)
        print("\n".join([f"{square.x * scale + 1} {square.y * scale + 1} {square.size*scale}" for square in self.squares]))

    def add_log(self, squares):
        self.log.append((tuple((square.x, square.y, square.size) for square in squares),
                         len(squares), self.count))


def find_free_point(square_map, size, x0, y0):
    for x in range(x0, size):
        for y in range(y0, size):
            if square_map[y][x] == 0:
                return x, y
        y0 = 0


def remove_square(square_map, square):
    for i in range(square.x, square.right+1):
        for j in range(square.y, square.bottom+1):
            square_map[j][i] = 0


def init_squares(squares, square_map, size):
    half = (size + 1) // 2
    small_half = size // 2
    squares.append(Square(0,0, half))
    squares.append(Square(0, half, small_half))
    squares.append(Square(half, 0, small_half))
    add_square(square_map, Square(0,0, half))
    add_square(square_map, Square(0, half, small_half))
    add_square(square_map, Square(half, 0, small_half))
    return half * half + small_half * small_half * 2


def greatest_divisor(n):
    divisor = 1
    for i in range(1, n//2+1):
        if n % i == 0:
            divisor = i
    return divisor


def backtrack(squares: list, square_map:list, count: int, filled_area: int, x0: int, y0: int, size: int, best: Solution):
    best.operation_count += 1
    x, y = find_free_point(square_map, size, x0, y0)
    max_size = min(size - x, size - y)
    for i in range(y, size + 1):
        if i == size:
            break
        if square_map[i][x] == 1:
            break
    max_size = min(max_size, i - y)
    for n in range(max_size, 0, -1):
        remaining_area = size * size - filled_area - n*n
        print(f"Попытка поставить квадрат на {x}, {y} размером {n}; ", end="")
        if remaining_area > 0:
            max_possible_size = min(size - x, size - y)
            min_squares_needed = remaining_area / (max_possible_size * max_possible_size)
            lower_bound = ceil(count + 1 + min_squares_needed)
            print(f"нижняя граница кол-ва квадратов для заполнения: {lower_bound}; ", end="")
            if lower_bound > best.count:
                print(f"больше {best.count}, пропускаем")
                continue

        new_square = Square(x, y, n)

        squares.append(new_square)
        add_square(square_map, new_square)
        if filled_area + n * n == size * size:
            print("квадрат заполнен; ", end='')
            if count + 1 < best.count:
                best.count = count + 1
                best.squares = squares.copy()
                best.solution_index = len(best.log)
                print(f"\nНовое решение ({count+1}):", ' | '.join([str(square) for square in squares]))
            best.add_log(squares)
            print("удаляем последний квадрат")
            squares.pop(-1)
            remove_square(square_map, new_square)
            best.add_log(squares)
            continue
        elif count + 1 < best.count:
            best.add_log(squares)
            print()
            backtrack(squares, square_map, count + 1, filled_area + n * n, x, y, size, best)
            print(f"Удаляем квадрат {new_square}")
            squares.pop(-1)
            best.add_log(squares)
        else:
            print("превышен минимум, возвращаемся")
        remove_square(square_map, new_square)


def add_square(square_map, square):
    for i in range(square.x, square.right+1):
        for j in range(square.y, square.bottom+1):
            square_map[j][i] = 1


def run_algorithm(n):
    scale = greatest_divisor(n)
    grid_size = n // scale
    squares = list()
    square_map = [[0] * grid_size for i in range(grid_size)]
    filled = init_squares(squares, square_map, grid_size)
    solution = Solution(grid_size * grid_size + 1, [], scale, grid_size)
    solution.add_log(squares)
    print("Начало, заполняем угол:", ' | '.join([str(square) for square in squares]))
    backtrack(squares, square_map, 3, filled, 0, 0, grid_size, solution)
    solution.print(scale)
    print("Backtrack function called:", solution.operation_count)
    return solution


window = TkWindow({"run_alg": run_algorithm})
window.mainloop()