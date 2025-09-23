import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sys import exit


class TkWindow(tk.Tk):
    def finish(self):
        self.destroy()
        exit()

    def __init__(self, functions):
        super().__init__()
        self.geometry("1100x600")
        self.resizable(False, False)
        self.solution = None
        self.current_step = -1
        self.params = dict()
        self.functions = functions
        self.finished = False
        frame = ttk.Frame(self)
        entry_frame = ttk.Frame(frame)
        ttk.Label(entry_frame, text="Размер квадрата").grid(padx=5, pady=5, column=0, row=0, sticky=tk.W)
        param1 = ttk.Entry(entry_frame, width=15)
        param1.grid(padx=5, pady=5, column=1, row=0, sticky=tk.W)
        entry_frame.grid(column=0, row=0, sticky=tk.W)
        self.param_fields = {"size": param1}

        ttk.Button(frame, text="Выполнить для заданного размера", command=self.execute_button).grid(padx=5, pady=5, column=0, row=1, sticky=tk.W)
        ttk.Button(frame, text="Показать решение", command=self.show_solution).grid(padx=5, pady=5, column=0, row=2, sticky=tk.W)
        ttk.Button(frame, text="Просмотр с начала", command=self.show_start).grid(padx=5, pady=5, column=0, row=3, sticky=tk.W)
        ttk.Button(frame, text="Просмотр с конца", command=self.show_end).grid(padx=5, pady=5, column=0, row=4, sticky=tk.W)
        ttk.Button(frame, text="Следующий шаг", command=self.step_forward).grid(padx=5, pady=5, column=0, row=5, sticky=tk.W)
        ttk.Button(frame, text="Предыдущий шаг", command=self.step_back).grid(padx=5, pady=5, column=0, row=6, sticky=tk.W)

        frame.grid(column=0, row=0)

        plot_frame = ttk.Frame(self)
        plot_frame.grid(padx=5, pady=5, column=1, row=0)

        figure1 = Figure(figsize=(6,6))
        plot1 = figure1.add_subplot(1, 1, 1)
        plot1.set_title("Текущее состояние алгоритма")
        ax1 = figure1.gca()
        ax1.xaxis.get_major_locator().set_params(integer=True)
        ax1.yaxis.get_major_locator().set_params(integer=True)
        self.canvas_process = FigureCanvasTkAgg(figure1, master=plot_frame)
        self.canvas_process.get_tk_widget().grid(row=0,column=0)

        self.info_text = ttk.Label(self, wraplength=250, text="")
        self.info_text.grid(row=0, column=2, padx=5, pady=5, sticky=tk.NSEW)

        self.protocol("WM_DELETE_WINDOW", self.finish)

    def update_params(self):
        for param in self.param_fields:
            self.params[param] = float(self.param_fields[param].get())

    def draw_solution(self, solution, size):
        ax = self.canvas_process.figure.gca()
        ax.cla()
        ax.set_title("Текущее состояние алгоритма")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylim(size, 0)
        ax.set_xlim(0, size)
        for square in solution:
            rect = Rectangle((square[0], square[1]), square[2], square[2], linewidth=1, edgecolor='black',
                             facecolor='lightblue')
            ax.add_patch(rect)
        self.canvas_process.draw()

    def execute_button(self):
        self.finished = False
        self.update_params()
        self.solution = self.functions["run_alg"](int(self.params["size"]))
        self.finished = True
        self.show_solution()

    def show_start(self):
        if not self.finished:
            return
        self.current_step = 0
        self.update_window()

    def show_end(self):
        if not self.finished:
            return
        self.current_step = len(self.solution.log)-1
        self.update_window()

    def step_forward(self):
        if not self.finished:
            return
        if self.current_step == len(self.solution.log) - 1 or self.current_step == -1:
            return
        self.current_step += 1
        self.update_window()

    def step_back(self):
        if not self.finished:
            return
        if self.current_step == 0:
            return
        self.current_step -= 1
        self.update_window()

    def show_solution(self):
        if not self.finished:
            return
        self.current_step = self.solution.solution_index
        self.update_window()

    def update_window(self):
        info = self.solution.log[self.current_step]
        text = (f"Заданный размер квадрата: {self.params["size"]}\n"
                f"Коэффициент масштаба: {self.solution.scale}\n"
                f"Выводится шаг: {self.current_step+1}/{len(self.solution.log)}\n"
                f"Минимальное количество квадратов на данном шаге: {info[2]}\n"
                f"Итоговое минимальное количество квадратов: {self.solution.count}\n"
                f"Количество операций (вызовов функции backtrack): {self.solution.operation_count}")
        self.draw_solution(info[0], self.solution.grid_size)
        self.info_text.config(text=text)
        self.update()

