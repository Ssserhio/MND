# Бурбело С. ІВ-91
from prettytable import PrettyTable 
from random import randint

# Почати відлік часу виконання програми:
start = timeit.default_timer()

# Оголошую змінні:
a0, a1, a2, a3 = 6, 4, 2, 5
n = 8 
x1, x2, x3 = [], [], []
y = []
min_y, max_y, x0 = int(), int(), int()
dx = float() 
y0 = float() 
xn1, xn2, xn3 = [], [], []
dx1, dx2, dx3 = int(), int(), int()
y_opt_list = []

# Генеруємо х1, х2, х3:
def generateX():
    for i in range(n):
        x1.append(randint(0,20))
        x2.append(randint(0,20))
        x3.append(randint(0,20))
    return x1, x2, x3 
generateX()

# Обчислюємо У:
def Y(x1, x2, x3):
    for j in range(n):
        y.append(a0 + a1*x1[j] + a2*x2[j] + a3*x3[j])
    return y
Y(x1, x2, x3)

# Знаходимо максимальне, мінімальне значення "У" та центр експеременту:
max_y, min_y= max(y), min(y)
max_x1, min_x1 = max(x1), min(x1)
max_x2, min_x2 = max(x2), min(x2)
max_x3, min_x3 = max(x3), min(x3)
x0 = (max_y + min_y)/2
x01 = (max_x1 + min_x1)/2
x02 = (max_x2 + min_x2)/2
x03 = (max_x3 + min_x3)/2

# Обчислюємо крок експеременту:
dx = x0 - min_y
dx1 = x01 - min_x1
dx2 = x02 - min_x2
dx3 = x03 - min_x3

# Виконуємо нормування:
for i in range(n):
    xn1.append(round(((x1[i] - x01)/dx1),3))
    xn2.append(round(((x2[i] - x02)/dx2),3))
    xn3.append(round(((x3[i] - x03)/dx3),3))

# У еталонне:
y0 = a0 + a1*x01 + a2*x02 + a3*x03

# Пошук точки в якій значення У найближче справа до У еталонного:
for i in range(len(y)):
    if y[i] > y0:
        y_opt_list.append(y[i])

def nearest_right(y, y0):
    return min(y, key=lambda x: abs(x-y0))
nearest_y = nearest_right(y_opt_list,y0)

# Закінчити відлік часу виконання програми:
stop = timeit.default_timer()

# Обчислюємо час виконання програми:
time = (stop - start)

# Вивід данних:
print("y = " + str(a0) + " + " + str(a1) + "*x1 + "+ str(a2) + "*x2 + " + str(a3) + "*x3")
num = [i for i in range(1,n+1)]
th = ["Num", "X1", "X2", "X3", "Y", "Xn1", "Xn2", "Xn3"]
columns = len(th)
table = PrettyTable(th)
for i in range(len(num)):
    td = [num[i], x1[i], x2[i], x3[i], y[i], xn1[i], xn2[i], xn3[i]]
    td_data = td[:]
    while td_data:
        table.add_row(td_data[:columns])
        td_data = td_data[columns:]
print(table)
print(f"| Y(еталонне) = ", y0)
print(f"| Y(оптимальне) = ", nearest_y)
print(f"Час роботи програми = ", time, "sec")
