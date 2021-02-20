from prettytable import PrettyTable 
from random import randint
import numpy as np
import math as m

n = 6 # кількість дослідів
x0 = 1  # додатковий нульовий фактор
y_max = (30 - 4) * 10
y_min = (20 - 4) * 10
x1_min, x1_max =  15, 45
x2_min, x2_max = -25, 10
xn = [[-1, -1], [1, -1], [-1, 1]]
x1, x2 = [-1, 1, -1], [-1, -1, 1]

y = [[],[],[]]
y1 = [randint(y_min, y_max) for j in range(3)]
y2 = [randint(y_min, y_max) for j in range(3)]
y3 = [randint(y_min, y_max) for j in range(3)]
y4 = [randint(y_min, y_max) for j in range(3)]
y5 = [randint(y_min, y_max) for j in range(3)]
y6 = [randint(y_min, y_max) for j in range(3)]
y[0] = [y1[0], y2[0], y3[0], y4[0], y5[0], y6[0]]
y[1] = [y1[1], y2[1], y3[1], y4[1], y5[1], y6[1]]
y[2] = [y1[2], y2[2], y3[2], y4[2], y5[2], y6[2]]

# ------------------------------------------------------------------------------------
# Перевіримо однорідність дисперсії за критерієм Романовського:
# ------------------------------------------------------------------------------------
# Знайдемо середнє значення функції відгуку в рядку: 
def averageY(y):
    average_y = []
    for i in range(len(y)):
        sum = 0
        for j in y[i]:
            sum += j
        average_y.append(sum / len(y[i]))
    return average_y

# Знайдемо дисперсії по рядках: 
def dispersion(y):
    dispersion = []
    for i in range(len(y)):
        sum = 0
        for j in y[i]:
            sum += (j - averageY(y)[i]) * (j - averageY(y)[i])
        dispersion.append(sum / len(y[i]))
    return dispersion

# Обчислимо основне відхилення:
mainErr = m.sqrt((2*(2*n-2))/(n*(n-4))) 

# Для кожної пари комбінацій u, v обчислимо Fuv: 
def fUV(u, v):
    if u >= v:
        return u / v
    else:
        return v / u
f_uv = []
f_uv.append(fUV(dispersion(y)[0], dispersion(y)[1]))
f_uv.append(fUV(dispersion(y)[2], dispersion(y)[0]))
f_uv.append(fUV(dispersion(y)[2], dispersion(y)[1]))

teta_uv = []
for i in range(3):
    teta_uv.append(((n-2)/n)*f_uv[i])

# Експериментальне значення критерію Романовського Ruv:
R_uv = []
for i in range(3):
    R_uv.append(abs(teta_uv[i]-1)/mainErr)

р = 0.99 # довірча ймовірність
R_kr = 2.16 # значення критерію Романовського за довірчої ймовірності 0.99:
for i in range(3):
    if R_uv[i] < R_kr:
        check_1 = "Дисперсія однорідна." 
    else:
        print('Помилка, повторіть експеримент')

# ------------------------------------------------------------------------------------
# Розрахунок нормованих коефіцієнтів рівняння регресії
# ------------------------------------------------------------------------------------
average_y = averageY(y)
x1, x2 = [-1, 1, -1], [-1, -1, 1]
mx1 = (x1[0] + x1[1] + x1[2]) / 3
mx2 = (x2[0] + x2[1] + x2[2]) / 3
my = (average_y[0] + average_y[1] + average_y[2]) / 3

a1 = (m.pow(x1[0],2) + m.pow(x1[1],2) + m.pow(x1[2],2))/3
a2 = (x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2])/3
a3 = (m.pow(x2[0],2) + m.pow(x2[1],2) + m.pow(x2[2],2))/3

a11 = (x1[0]*average_y[0] + x1[1]*average_y[1] + x1[2]*average_y[2])/3
a22 = (x2[0]*average_y[0] + x2[1]*average_y[1] + x2[2]*average_y[2])/3

def discriminant(x11, x12, x13, x21, x22, x23, x31, x32, x33):
    discriminant = x11*x22*x33 + x12*x23*x31 + x32*x21*x13 - x13*x22*x31 - x32*x23*x11 - x12*x21*x33
    return discriminant

b0 = discriminant(my,mx1,mx2,a11,a1,a2,a22,a2,a3)/discriminant(1,mx1,mx2,mx1,a1,a2,mx2,a2,a3)
b1 = discriminant(1,my,mx2,mx1,a11,a2,mx2,a22,a3)/discriminant(1,mx1,mx2,mx1,a1,a2,mx2,a2,a3)
b2 = discriminant(1,mx1,my,mx1,a1,a11,mx2,a2,a22)/discriminant(1,mx1,mx2,mx1,a1,a2,mx2,a2,a3)

# Нормоване рівняння регресії:
y_norm = "y = " + str(round(b0, 3)) + " + " + str(round(b1, 3)) + "*x1 + " + str(round(b2, 3)) + "*x2"
y_norm1 = b0 + b1*x1[0] + b2*x2[0]
y_norm2 = b0 + b1*x1[1] + b2*x2[1]
y_norm3 = b0 + b1*x1[2] + b2*x2[2]
y_norms = [y_norm1, y_norm2, y_norm3]

# Зробимо перевірку: 
for i in range(3):
    if round(y_norms[i],5) == round(average_y[i],5):
        check_2 = "Результат збігається з середніми значеннями Yj." 
    else: 
        check_2 = "Результат НЕ збігається з середніми значеннями Yj." 

# ------------------------------------------------------------------------------------
# Проведемо натуралізацію коефіцієнтів:
# ------------------------------------------------------------------------------------
delta_x1 = abs(x1_max - x1_min)/2
delta_x2 = abs(x2_max - x2_min)/2
x10 = (x1_max + x1_min)/2
x20 = (x2_max + x2_min)/2

a_0 = b0 - (b1*x10 / delta_x1) - (b2*x20 / delta_x2)
a_1 = b1/delta_x1
a_2 = b2/delta_x2

# Запишемо натуралізоване рівняння регресії: 
y_nut1 = a_0 + a_1*x1_min + a_2*x2_min
y_nut2 = a_0 + a_1*x1_max + a_2*x2_min
y_nut3 = a_0 + a_1*x1_min + a_2*x2_max
y_nut = [y_nut1, y_nut2, y_nut3]

# Зробимо перевірку: 
for i in range(3):
    if round(y_nut[i],5) == round(average_y[i],5):
        check_3 = "Коефіцієнти натуралізованого рівняння регресії вірні." 
    else: 
        check_3 = "Коефіцієнти натуралізованого рівняння не збігаються." 
# ------------------------------------------------------------------------------------
# Вивід даних: 
# ------------------------------------------------------------------------------------
print("Лінійне рівняння регресії : y = b0 + b1*x1 + b2*x2")
th = ["X1", "X2", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6"]
columns = len(th)
rows = len(xn)
table = PrettyTable(th)
table.title = 'Нормована матриця планування експерименту'
for i in range(rows):
    td = [xn[i][0], xn[i][1], y1[i], y2[i], y3[i], y4[i], y5[i], y6[i]]
    td_data = td[:]
    while td_data:
        table.add_row(td_data[:columns])
        td_data = td_data[columns:]
print(table)
print("Cередній Y:\n", round(average_y[0],3), "\n", round(average_y[1],3), "\n", round(average_y[2],3))
print("--------------------------------------------------------------")
print("Експериментальні значення критерію Романовського:\n", \
     round(R_uv[0], 4), "\n", round(R_uv[1], 4), "\n", round(R_uv[2], 4))
print(check_1)
print("------------------------------------------------------------------")
print("Нормоване рівняння регресії: \n" + y_norm)
print("Нормований Y: \n", round(y_norms[0],3), "\n", round(y_norms[1],3), "\n", round(y_norms[2],3))
print(check_2)
print("------------------------------------------------------------------")
print("Натуралізоване рівняння регресії: \nу = a0 + a1*x1 + a2*x2 = "\
        + str(round(a_0,3)) + " + " + str(round(a_1,3)) + "*x1 + " + str(round(a_2,3)) + "*x2")
print("Натуралізований Y:\n", round(y_nut[0],3), "\n", round(y_nut[1],3), "\n", round(y_nut[2],3))
print(check_3)