from prettytable import PrettyTable 
from random import randint
from numpy.linalg import solve
from _pydecimal import Decimal
from scipy.stats import f, t
import math

class Experiment:

    def __init__(self, n, m):

        self.n = n
        self.m = m

        self.f1 = self.m - 1
        self.f2 = self.n
        self.f3 = self.f1*self.f2

        self.p = 0.95
        self.q = 1 - self.p

        self.N = [i+1 for i in range(self.n+1)]

        self.Kohren = False
        self.Fisher = False

        self.x1_min = -20
        self.x1_max = 30
        self.x2_min = -25
        self.x2_max = 10
        self.x3_min = -25
        self.x3_max = -20
        self.x01 = (self.x1_max + self.x1_min) / 2
        self.x02 = (self.x2_max + self.x2_min) / 2
        self.x03 = (self.x3_max + self.x3_min) / 2
        self.delta_x1 = self.x1_max - self.x01
        self.delta_x2 = self.x2_max - self.x02
        self.delta_x3 = self.x3_max - self.x03

            
        self.matrix_pfe = [[-1, -1, -1, +1, +1, +1, -1, +1, +1, +1],
                            [-1, -1, +1, +1, -1, -1, +1, +1, +1, +1],
                            [-1, +1, -1, -1, +1, -1, +1, +1, +1, +1],
                            [-1, +1, +1, -1, -1, +1, -1, +1, +1, +1],
                            [+1, -1, -1, -1, -1, +1, +1, +1, +1, +1],
                            [+1, -1, +1, -1, +1, -1, -1, +1, +1, +1],
                            [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
                            [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
                            [-1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
                            [+1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
                            [0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
                            [0, +1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
                            [0, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929],
                            [0, 0, +1.73, 0, 0, 0, 0, 0, 0, 2.9929],
                            [0, 0,   0,   0, 0, 0, 0, 0, 0,   0   ]]

        self.matrix_x = [[] for x in range(self.n)]
        for i in range(len(self.matrix_x)):
            if i < 8:
                self.x_1 = self.x1_min if self.matrix_pfe[i][0] == -1 else self.x1_max
                self.x_2 = self.x2_min if self.matrix_pfe[i][1] == -1 else self.x2_max
                self.x_3 = self.x3_min if self.matrix_pfe[i][2] == -1 else self.x3_max
            else:
                self.x_lst = self.x(self.matrix_pfe[i][0], self.matrix_pfe[i][1], self.matrix_pfe[i][2])
                self.x_1, self.x_2, self.x_3 = self.x_lst
            self.matrix_x[i] = [self.x_1, self.x_2, self.x_3, self.x_1 * self.x_2, self.x_1 * self.x_3, self.x_2 * \
                 self.x_3, self.x_1 * self.x_2 * self.x_3, self.x_1 ** 2, self.x_2 ** 2, self.x_3 ** 2]

        self.Result(self.n, self.m, self.f1, self.f2, self.f3, self.q, self.matrix_x)


    def generate_matrix(self, matrix_x):

        def f(X1, X2, X3):
            f = 8.7 + 4.3 * X1 + 1.2 * X2 + 2.2 * X3 + 0.4 * X1 * X1 + 1.0 * X2 * X2 + 6.4 * X3 * X3 + 1.1 * X1 * X2 + \
                0.1 * X1 * X3 + 9.2 * X2 * X3 + 1.2 * X1 * X2 * X3 + randint(0, 10) - 5
            return f

        matrix_with_y = [[f(matrix_x[j][0], matrix_x[j][1], matrix_x[j][2]) for i in range(self.m)] for j in range(self.n)]
        return matrix_with_y


    def x(self, l1, l2, l3):
        x_1 = l1 * self.delta_x1 + self.x01
        x_2 = l2 * self.delta_x2 + self.x02
        x_3 = l3 * self.delta_x3 + self.x03
        return [x_1, x_2, x_3]


    def find_average(self, lst, orientation):
        average = []
        if orientation == 1:
            for rows in range(len(lst)):
                average.append(sum(lst[rows]) / len(lst[rows]))
        else:
            for column in range(len(lst[0])):
                number_lst = []
                for rows in range(len(lst)):
                    number_lst.append(lst[rows][column])
                average.append(sum(number_lst) / len(number_lst))
        return average


    def find_known(self, number):
        need_a = 0
        for j in range(self.n):
            need_a += self.average_y[j] * self.matrix_x[j][number - 1] / 15
        return need_a


    def solve(self, lst_1, lst_2):
        solver = solve(lst_1, lst_2)
        return solver


    def check_result(self, b_lst, k):
        y_i = b_lst[0] + b_lst[1] * self.matrix[k][0] + b_lst[2] * self.matrix[k][1] + b_lst[3] * self.matrix[k][2] + \
            b_lst[4] * self.matrix[k][3] + b_lst[5] * self.matrix[k][4] + b_lst[6] * self.matrix[k][5] + b_lst[7] * self.matrix[k][6] + \
            b_lst[8] * self.matrix[k][7] + b_lst[9] * self.matrix[k][8] + b_lst[10] * self.matrix[k][9]
        return y_i
    
    # -------------------------------------------------------
    # Перевірка однорідності дисперсії за критерієм Кохрена: 
    # -------------------------------------------------------
    def get_cohren_value(self, size_of_selections, qty_of_selections, significance):
        size_of_selections += 1
        partResult1 = significance / (size_of_selections - 1)
        params = [partResult1, qty_of_selections, (size_of_selections - 1 - 1) * qty_of_selections]
        fisher = f.isf(*params)
        result = fisher / (fisher + (size_of_selections - 1 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    # -------------------------------------------------------
    # Перевірка однорідності дисперсії за критерієм Стьюдента: 
    # -------------------------------------------------------
    def student_test(self, b_lst, number_x=10):
        dispersion_b = math.sqrt(self.dispersion_b2)
        for column in range(number_x + 1):
            t_practice = 0
            t_theoretical = self.get_student_value(self.f3, self.q)
            for row in range(self.n):
                if column == 0:
                    t_practice += self.average_y[row] / self.n
                else:
                    t_practice += self.average_y[row] * self.matrix_pfe[row][column - 1]
            if math.fabs(t_practice / dispersion_b) < t_theoretical:
                b_lst[column] = 0
        return b_lst

    def get_student_value(self, f3, significance):
        return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()

    # -------------------------------------------------------
    # Перевірка однорідності дисперсії за критерієм Фішера: 
    # -------------------------------------------------------
    def fisher_test(self):
        dispersion_ad = 0
        for row in range(len(self.average_y)):
            dispersion_ad += (self.m * (self.average_y[row] - self.check_result(self.student_lst, row))) / (self.n - self.d)
        self.F_practice = dispersion_ad / self.dispersion_b2
        F_theoretical = self.get_fisher_value(self.q)
        return self.F_practice < F_theoretical

    def get_fisher_value(self, significance):
        return Decimal(abs(f.isf(significance, self.f4, self.f3))).quantize(Decimal('.0001')).__float__()


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Вивід даних:
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def Result(self, n, m, f1, f2, f3, q, matrix_x):

        while not self.Fisher:

            matrix_y = self.generate_matrix(matrix_x)
            average_x = self.find_average(matrix_x, 0)
            self.average_y = self.find_average(matrix_y, 1)
            self.matrix = [(matrix_x[i] + matrix_y[i]) for i in range(n)]
            mx_i = average_x
            my = sum(self.average_y) / 15

            dispersion_y = [0.0 for x in range(n)]
            for i in range(n):
                dispersion_i = 0
                for j in range(m):
                    dispersion_i += (matrix_y[i][j] - self.average_y[i]) ** 2
                dispersion_y.append(dispersion_i / (m - 1))

            print("\nx1min = 15.0 x1max = 25.0\nx2min = -25.0 x2max = 10.0\nx3min = 45.0 x3max = 50.0")
            
            print("\nf(x1, x2, x3) = 8,7+4,3*x1+1,2*x2+2,2*x3+0,4*x1*x1+1,0*x2*x2+6,4*x3*x3+1,1*x1*x2+0,1*x1*x3+9,2*x2*x3+1,2*x1*x2*x3")
            print("\nРівняння регресії: y = b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1*x1+b22*x2*x2+b33*x3*x3\n")

            th = ["N", "x1", "x2", "x3", "x1*x2", "x1*x3", "x2*x3", "x1*x2*x3", "x1^2", "x2^2", "x3^2"] 
            columns = len(th)
            table = PrettyTable(th)
            table.title = "Матриця планування експерименту для РЦКП з нормованими значеннями факторів:"
            for i in range(self.n):
                td = [self.N[i], self.matrix_pfe[i][0], self.matrix_pfe[i][1], self.matrix_pfe[i][2], \
                    round(self.matrix_pfe[i][3],3), round(self.matrix_pfe[i][4],3), round(self.matrix_pfe[i][5],3), round(self.matrix_pfe[i][6],3), \
                        round(self.matrix_pfe[i][7],3), round(self.matrix_pfe[i][8],3), round(self.matrix_pfe[i][9],3)  ]
                td_data = td[:]
                while td_data:
                    table.add_row(td_data[:columns])
                    td_data = td_data[columns:]
            print(table)

            th = ["N", "x1", "x2", "x3", "x1*x2", "x1*x3", "x2*x3", "x1*x2*x3", "x1^2", "x2^2", "x3^2"] 
            th.append("<y>")
            columns = len(th)
            table = PrettyTable(th)
            table.title = "Матриця планування експерименту для РЦКП з натуральними значеннями факторів: "
            for i in range(self.n):
                td = [self.N[i], self.matrix[i][0], self.matrix[i][1], self.matrix[i][2], \
                    round(self.matrix[i][3],3), round(self.matrix[i][4],3), round(self.matrix[i][5],3), round(self.matrix[i][6],3), \
                        round(self.matrix[i][7],3), round(self.matrix[i][8],3), round(self.matrix[i][9],3)  ]
                td.append(round(self.average_y[i],2))
                td_data = td[:]
                while td_data:
                    table.add_row(td_data[:columns])
                    td_data = td_data[columns:]
            print(table)

            def a(first, second):
                need_a = 0
                for j in range(n):
                    need_a += matrix_x[j][first - 1] * matrix_x[j][second - 1] / n
                return need_a

            koef = [
                [1, mx_i[0], mx_i[1], mx_i[2], mx_i[3], mx_i[4], mx_i[5], mx_i[6], mx_i[7], mx_i[8], mx_i[9]],
                [mx_i[0], a(1, 1), a(1, 2), a(1, 3), a(1, 4), a(1, 5), a(1, 6), a(1, 7), a(1, 8), a(1, 9), a(1, 10)],
                [mx_i[1], a(2, 1), a(2, 2), a(2, 3), a(2, 4), a(2, 5), a(2, 6), a(2, 7), a(2, 8), a(2, 9), a(2, 10)],
                [mx_i[2], a(3, 1), a(3, 2), a(3, 3), a(3, 4), a(3, 5), a(3, 6), a(3, 7), a(3, 8), a(3, 9), a(3, 10)],
                [mx_i[3], a(4, 1), a(4, 2), a(4, 3), a(4, 4), a(4, 5), a(4, 6), a(4, 7), a(4, 8), a(4, 9), a(4, 10)],
                [mx_i[4], a(5, 1), a(5, 2), a(5, 3), a(5, 4), a(5, 5), a(5, 6), a(5, 7), a(5, 8), a(5, 9), a(5, 10)],
                [mx_i[5], a(6, 1), a(6, 2), a(6, 3), a(6, 4), a(6, 5), a(6, 6), a(6, 7), a(6, 8), a(6, 9), a(6, 10)],
                [mx_i[6], a(7, 1), a(7, 2), a(7, 3), a(7, 4), a(7, 5), a(7, 6), a(7, 7), a(7, 8), a(7, 9), a(7, 10)],
                [mx_i[7], a(8, 1), a(8, 2), a(8, 3), a(8, 4), a(8, 5), a(8, 6), a(8, 7), a(8, 8), a(8, 9), a(8, 10)],
                [mx_i[8], a(9, 1), a(9, 2), a(9, 3), a(9, 4), a(9, 5), a(9, 6), a(9, 7), a(9, 8), a(9, 9), a(9, 10)],
                [mx_i[9], a(10, 1), a(10, 2), a(10, 3), a(10, 4), a(10, 5), a(10, 6), a(10, 7), a(10, 8), a(10, 9), a(10, 10)]
            ]
            known = [my, self.find_known(1), self.find_known(2), self.find_known(3), self.find_known(4), self.find_known(5), \
                 self.find_known(6), self.find_known(7), self.find_known(8), self.find_known(9), self.find_known(10)]

            beta = self.solve(koef, known)
            print("\nОтримане рівняння регресії: y = {:.3f} + {:.3f} * x1 + {:.3f} * x2 + {:.3f} * x3 + {:.3f} * x1x2 + {:.3f} * x1x3 + {:.3f} * x2x3"
                "+ {:.3f} * x1x2x3 + {:.3f} * x1^2 + {:.3f} * x2^2 + {:.3f} * x3^2 \nПеревірка"
                .format(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], beta[6], beta[7], beta[8], beta[9], beta[10]))
            for i in range(n):
                print("ŷ{} = {:.3f} ≈ {:.3f}".format((i + 1), self.check_result(beta, i), self.average_y[i]))
            
            while not self.Kohren:

                Gp = max(dispersion_y) / sum(dispersion_y)
                print("\nКритерій Кохрена: Gp = {}".format(Gp))
                Gt = self.get_cohren_value(f2, f1, q)
                if Gt > Gp:
                    print("Дисперсія однорідна при рівні значимості {:.2f}.\n".format(q))
                    self.Kohren = True
                else:
                    print("Дисперсія не однорідна при рівні значимості {:.2f}! Збільшуємо m.\n".format(q))
                    self.m += 1

            self.dispersion_b2 = sum(dispersion_y) / (n * n * m)
            self.student_lst = list(self.student_test(beta))
            print("Отримане рівняння регресії з урахуванням критерія Стьюдента: y = {:.3f} + {:.3f} * x1 + {:.3f} * x2 + {:.3f} * x3 + {:.3f} * x1x2 + {:.3f} * x1x3 + {:.3f} * x2x3"
                "+ {:.3f} * x1x2x3 + {:.3f} * x1^2 + {:.3f} * x2^2 + {:.3f} * x3^2 \nПеревірка"
                .format(self.student_lst[0], self.student_lst[1], self.student_lst[2], self.student_lst[3], self.student_lst[4], self.student_lst[5],
                        self.student_lst[6], self.student_lst[7], self.student_lst[8], self.student_lst[9], self.student_lst[10]))
            for i in range(n):
                print("ŷ{} = {:.3f} ≈ {:.3f}".format((i + 1), self.check_result(self.student_lst, i), self.average_y[i]))

            self.d = 11 - self.student_lst.count(0)
            self.f4 = self.n - self.d 
            self.fisher_test()
            print("\nКритерій Фішера: Fp = {}".format(self.F_practice))
            if self.fisher_test():
                print("Рівняння регресії адекватне  оригіналу")
                self.Fisher = True
            else:
                print("Рівняння регресії неадекватне  оригіналу\n\t Проводимо експеремент повторно")

if __name__ == '__main__':
    Experiment(15, 3)