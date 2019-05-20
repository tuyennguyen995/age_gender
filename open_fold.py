import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.getcwd()
print(dir_path)
os.remove(dir_path + '/dataset/data.txt')
for n in range(5):
#Nối các fold data
    file_r = open('dataset/fold_'+str(n)+'_data.txt')
    file_w = open('dataset/data.txt', 'a+')
    data_r = file_r.read()
    data_w = file_w.write(data_r)
    file_r.close()
    file_w.close()

#Thống kê
data_r = open('dataset/data.txt')
row = csv.reader(data_r, delimiter='\t')

a = np.zeros((4, 10), int)
for x in row:
    if x[3] == '(0, 2)':
        if x[4] == 'm':
            a[0][0] += 1
        elif x[4] == 'f':
            a[1][0] += 1
        else:
            a[2][0] += 1
    elif x[3] == '(4, 6)':
        if x[4] == 'm':
            a[0][1] += 1
        elif x[4] == 'f':
            a[1][1] += 1
        else:
            a[2][1] += 1
    elif x[3] == '(8, 12)':
        if x[4] == 'm':
            a[0][2] += 1
        elif x[4] == 'f':
            a[1][2] += 1
        else:
            a[2][2] += 1
    elif x[3] == '(15, 20)':
        if x[4] == 'm':
            a[0][3] += 1
        elif x[4] == 'f':
            a[1][3] += 1
        else:
            a[2][3] += 1
    elif x[3] == '(25, 32)':
        if x[4] == 'm':
            a[0][4] += 1
        elif x[4] == 'f':
            a[1][4] +=  1
        else:
            a[2][4] += 1
    elif x[3] == '(38, 43)':
        if x[4] == 'm':
            a[0][5] += 1
        elif x[4] == 'f':
            a[1][5] += 1
        else:
            a[2][5] += 1
    elif x[3] == '(48, 53)':
        if x[4] == 'm':
            a[0][6] += 1
        elif x[4] == 'f':
            a[1][6] +=  1
        else:
            a[2][6] += 1
    elif x[3] == '(60, 100)':
        if x[4] == 'm':
            a[0][7] += 1
        elif x[4] == 'f':
            a[1][7] +=  1
        else:
            a[2][7] += 1
    else:
        if x[4] == 'm':
            a[0][8] += 1
        elif x[4] == 'f':
            a[1][8] +=  1
        else:
            a[2][8] += 1
data_r.close()
for i in range(9):
    temp = 0
    for j in range(3):
        temp += a[j][i]
    a[3][i] = temp
for i in range(4):
    temp = 0
    for j in range(9):
        temp += a[i][j]
    a[i][9] = temp

print('Tổng số dòng là: '+str(a[3][9]))
print('--------------------------------')
print(a)
print('--------------------------------')

#Vẽ biểu đồ
#data_r = open('dataset/data.txt')
#row = csv.reader(data_r, delimiter='\t')
#for x in row:
    #y = np.array(['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100'])
#    plt.scatter(x[3], x[4])
#plt.show()
#data_r.close()