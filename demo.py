#!/usr/bin/env python

# -*- coding: utf-8 -*-

'''
@Author  :   Tomatohust
@Contact :   pengc.wang@foxmail.com
@Software:   Pycharm2021.3 Python3.9.6
@File    :   main.py
@Time    :   2021-8-12
@Desc    :   This is a demo of the Wolf-Pack-Algorithm. The proposed algorithm is applied to 12 typical
             complex function optimization pronlems.
'''


import time
import numpy as np
import random
from math import sin, cos, pi, floor, sqrt, exp
from matplotlib import pyplot as plt


class BenchMarkFunc:  # Define function classes, each method corresponding to the typlical function.
    def F1(self, x, y):
        f = -np.cos(x) * np.cos(y) * np.exp(-(x - pi) ** 2 - (y - pi) ** 2)
        return f

    def F2(self, x, y):
        f = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return f

    # def F3(self, x):
    #    f = sum(100 * (x(2:m) - (x(1:m-1). ^ 2)).^ 2 + (x(1:m-1) - 1).^ 2)
    #    return f

    # def F4(self, x):
    #     f = 100 * (x(1). ^ 2 - x(2)). ^ 2 + (x(1) - 1). ^ 2 + (x(3) - 1). ^ 2 + 90 * (x(3). ^ 2 - x(4)). ^ 2 + 10.1 * (
    #                 (x(2) - 1). ^ 2 + (x(4) - 1). ^ 2) + 19.8 * (x(2) - 1) * (x(4) - 1)

    def F5(self, x, y):
        m = x.shape[1]
        f = np.sum((x - 1) ** 2) - np.sum(x[0, 1:m] * x[0, 0:m - 1])
        return f

    def F6(self, x, y):
        f = 30 + np.floor(x).sum()
        return f

    def F8(self, x, y):
        m = np.arange(0, x.shape[1], 1)
        f = np.dot(x, np.multiply(x, m).T)
        return float(f)

    def F10(self, x, y):
        f = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        return f

    def F11(self, x, y):
        f = x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * pi * x) - 0.4 * np.cos(4 * pi * y) + 0.7
        return f

    def F14(self, x, y):
        f = x ** 2 + y ** 2 + 25 * ((np.sin(x)) ** 2 + (np.sin(y)) ** 2)
        return f

    def F15(self, x, y):
        f = 0.5 + ((np.sin(np.sqrt(x ** 2 + y ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
        return f

    def F16(self, x, y):
        f = 4 * (x ** 2) - 2.1 * (x ** 4) + (x ** 6) / 3 + x * y - 4 * (y ** 2) + 4 * (y ** 4)
        return f

    def F17(self, x, y):
        f = x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * pi * x + 4 * pi * y) + 0.3
        return f

    def F18(self, x, y):
        f = -(np.sin(np.sqrt(x ** 2 + y ** 2))/ np.sqrt(x ** 2 + y ** 2) + np.exp(
            (np.cos(2 * pi * x) + np.cos(2 * pi * y)) / 2) - 0.71289)
        return f


def simulate_fc(POS, func):  # Prey odor concentration estimation function
    n = POS.shape[0]
    Y = np.zeros((n, 1))
    w = POS.shape[1]
    for q in range(n):
        if w == 2:
            Y[q] = func(POS[q][0], POS[q][1])
        else:
            Y[q] = func(POS[q].reshape(1, -1), None)
    Y = -Y
    return Y


def paint3d(func, limit):  # Draws 3D surface of two-dimensional functions
    ax3 = plt.axes(projection='3d')
    xx = np.arange(-limit, limit, 0.1)
    yy = np.arange(-limit, limit, 0.1)
    X, Y = np.meshgrid(xx, yy)
    Z = func(X, Y)
    # ax3.plot_surface(X, Y, Z, cmap='rainbow')
    ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()
    return ax3


def compute_dist(pos1, pos2):  # Coordinate distance calculation, Manhattan distance.
    dist = 0
    w = pos1.shape[1]
    for j in range(w):
        dist = dist + abs(pos1.reshape(1, -1)[0][j] - pos2.reshape(1, -1)[0][j])
    return dist


def wolf_sort(POS, Y):  # Wolf pack sorting function
    if POS.shape[0] == Y.shape[0]:
        argue = np.hstack((POS, Y))
    else:
        print("ERROR")
        return -1
    temp = argue[np.lexsort(-argue.T)]
    return temp


def get_func_details(f_name):   # Get more details of typical complex functions
    BMF = BenchMarkFunc()
    limit = w = extrenum = 0
    func = None
    if f_name == 'F1':  # Easom function
        func = BMF.F1
        limit = 10
        w = 2
        extrenum = -1

    elif f_name == 'F2':  # Matyas function
        func = BMF.F2
        limit = 10
        w = 2
        extrenum = 0

    elif f_name == 'F3':  # Rosenbrock function
        func = BMF.F3
        limit = 2.048
        w = 2
        extrenum = 0

    elif f_name == 'F4':  # Colville function
        func = BMF.F4
        limit = 10
        w = 4
        extrenum = 0

    elif f_name == 'F5':  # Trid6 function
        func = BMF.F5
        limit = 36
        w = 6
        extrenum = -50

    elif f_name == 'F6':  # Stepint function
        func = BMF.F6
        limit = 5.12
        w = 5
        extrenum = 0

    elif f_name == 'F7':  # Step function
        func = BMF.F7
        limit = 100
        w = 30
        extrenum = 0

    elif f_name == 'F8':  # Sumsquares function
        func = BMF.F8
        limit = 10
        w = 10
        extrenum = 0

    elif f_name == 'F9':  # Sphere function
        func = BMF.F9
        limit = 1.5
        w = 200
        extrenum = 0

    elif f_name == 'F10':  # Booth function
        func = BMF.F10
        limit = 10
        w = 2
        extrenum = 0

    elif f_name == 'F11':  # Bohachevsky1 function
        func = BMF.F11
        limit = 100
        w = 2
        extrenum = 0

    elif f_name == 'F12':  # Rastrigin function
        func = BMF.F12
        limit = 5.12
        w = 30
        extrenum = 0

    elif f_name == 'F13':  # Quadric function
        func = BMF.F13
        limit = 30
        w = 100
        extrenum = 0

    elif f_name == 'F14':  # Eggcrate function
        func = BMF.F14
        limit = 2 * pi
        w = 2
        extrenum = 0

    elif f_name == 'F15':  # Schaffer function
        func = BMF.F15
        limit = 100
        w = 2
        extrenum = 0

    elif f_name == 'F16':  # Six Hump Camel Back function
        func = BMF.F16
        limit = 5
        w = 2
        extrenum = -1.0316

    elif f_name == 'F17':  # Bohachevsky1 function
        func = BMF.F17
        limit = 100
        w = 2
        extrenum = 0

    elif f_name == 'F18':  # Bridge function
        func = BMF.F18
        limit = 1.5
        w = 2
        extrenum = -3.0054

    elif f_name == 'F19':  # Griewank function
        func = BMF.F19
        limit = 600
        w = 100
        extrenum = 0

    elif f_name == 'F20':  # Ackley function
        func = BMF.F20
        limit = 32
        w = 200
        extrenum = 0

    return w, limit, extrenum, func


def BV_beleaguer(pos, y, step, pos_head, func):  # Wolf-beleaguer behavior
    pos_raw = pos.copy()
    y_raw = y.copy()
    for k in range(pos.shape[1]):
        r = random.random() * (-2) + 1  # Generates a random number within [-1, 1]
        pos[0][k] = pos[0][k] + r * step * abs(pos_head[0][k] - pos[0][k])  # Attack in the direction of the head wolf
    y = simulate_fc(pos, func)  # Calculate the odor-concentration again
    if y_raw >= y:  # Only keep the better attemp, otherwise go back to the original position
        pos = pos_raw
        y = y_raw
    return pos, y


def BV_scout(POS, Y, pos_head, y_head, step_sc, sc_max, dir_num, func):  # Wolf-scout behavior
    n = POS.shape[0]  # The number of artificial wolves
    w = POS.shape[1]  # Coding width
    POS_temp = np.zeros((dir_num, w))
    for j in range(n):
        for t in range(sc_max):
            if Y[j] > y_head:  # If find a better Wolf, then use it to update the head wolf
                pos_c = pos_head.copy()  # Copy the information of head wolf
                y_c = y_head.copy()
                pos_head = POS[j].copy().reshape(1, -1)
                y_head = Y[j].copy()
                POS[j] = pos_c
                Y[j] = y_c
                break
            else:
                for p in range(dir_num):  # Calculate the odor-concentration of every direction
                    POS_temp[p] = POS[j].reshape(1, -1) + step_sc * sin(2 * pi * p / dir_num)
                Y_temp = simulate_fc(POS_temp, func)
                dir_temp = wolf_sort(POS_temp, Y_temp)
                if dir_temp[0][w] > Y[j]:  # If find a better Wolf, then use it to update the head wolf
                    POS[j] = dir_temp[0, :w].reshape(1, -1)
                    Y[j] = dir_temp[0][w]
    return POS, Y, pos_head, y_head


def BV_update(POS, pos_head, oust_num, func):  # Update the wolf pack
    n = POS.shape[0]  # The number of artificial wolves
    w = POS.shape[1]  # Coding width
    POS_new = np.zeros((oust_num, w))
    Y = simulate_fc(POS, func)
    temp = wolf_sort(POS, Y)  # The pack is sorted to weed out the worst artificial wolves
    POS = temp[:n - oust_num, :w]
    for j in range(oust_num):  # Spawn new artificial wolves to replenish the ousted
        POS_new[j] = pos_head * (0.1 * (-2 * random.random() + 1) + 1)
    POS = np.vstack((POS, POS_new))  # Merge to form a new generation of wolve pack
    return POS


def BV_summon(POS, Y, pos_head, y_head, step_rn, step_bl, dist_min, func):  # Wolf-summon behavior
    n = POS.shape[0]  # The number of artificial wolves
    w = POS.shape[1]  # Coding width
    cnt = 0  # just a counter
    while cnt < n:
        f_clc = dist_c = 0  # A flag to get rid of shock
        head_flag = 0
        dist = compute_dist(POS[cnt].reshape(1, -1), pos_head)  # Calculate the distance from the head wolf
        while dist > dist_min:  # The Wolf is longer than the beleaguer distance, continues running
            for k in range(w):
                POS[cnt][k] = POS[cnt][k] + step_rn * (pos_head[0][k] - POS[cnt][k]) / abs(pos_head[0][k] - POS[cnt][k])
            Y[cnt] = simulate_fc(POS[cnt].reshape(1, -1), func)
            if Y[cnt] > y_head:  # If find a better Wolf, then use it to update the head wolf
                pos_c = pos_head.copy()  # Copy the information of head wolf
                y_c = y_head.copy()
                pos_head = POS[cnt].copy().reshape(1, -1)
                y_head = Y[cnt].copy()
                POS[cnt] = pos_c
                Y[cnt] = y_c
                head_flag = 1
                break
            dist = compute_dist(POS[cnt].reshape(1, -1), pos_head)  # Calculate the distance again

            # This part of code can get rid of 'shock'
            if dist < 1:
                a = 0
                pass
            if f_clc == 0:
                dist_c = dist.copy()
                f_clc = f_clc + 1
            elif f_clc == 2:
                if dist == dist_c:
                    f_clc = 0
                    break
                else:
                    f_clc = 0
                    pass
            else:
                f_clc = f_clc + 1

        # Endding of running, beleaguering starts
        if head_flag == 1:
            cnt = -1
            pass
        else:
            [POS[cnt], y_c] = BV_beleaguer(POS[cnt].reshape(1, -1), Y[cnt], step_bl, pos_head, func)
        cnt = cnt + 1
    return POS, Y, pos_head, y_head


def WPA_MAIN(f_name):  # WPA main function
    # Definition of parameter values which required
    # Note: the value of the Δ marker should be adjusted with the selected standard function
    n = 50  # The number of artificial wolves
    w = 0  # Δ Coding width
    a = 2  # Explore wolf scale factor
    step = 0.1  # Step length factor
    itr_num = 2000  # Maximum iteration
    cnt_num = 100  # Times the algorithm is repeated
    up_rate = 6  # Renewal factor
    extrenum = 0  # Δ Theoretical solution
    limit = 0  # Δ The absolute value of the upper and lower limits of the variable value
    dist_min = 0.1  # Distance factor of summon behavior
    step_param = 10  # Step differential factor
    sc_max = 10  # Maximun scout times
    f = 8  # Direction factor(determine the number of scouting directions)
    gate = 0.001  # The threshold of the optimal solution

    # Variable initialization
    S_num = random.randint(int(n / (a + 1)), int(n / a))  # Random the num of explore-wolfs
    M_num = n - S_num - 1  # Determine the num of fierce-wolfs
    POS = np.zeros((n, w))  # A matrix to storage the location of the whole wolf pack
    Y = np.zeros((n, 1))  # Odor concentration
    pos_head = np.zeros((1, w))  # Head wolf position
    y_head = 0
    step_sc = step / step_param  # scouting step length
    step_rn = step  # running step length
    step_bl = step / (2 * step_param)   # beleaguering step length
    dir_num = floor(random.random() * (f - 2 * f) + 2 * f)  # Determine the number of scouting directions
    oust_num = 0  # The num of ousted
    func = None  # Objective function expression

    suc_cnt = 0  # Success counter
    process_time = 0  # Timer
    total_time = 0  # Timer globe
    RESULT = []  # Store the result

    [w, limit, extrenum, func] = get_func_details(f_name)
    # print(func(np.array([1, 2, 3]).reshape(1, -1), None))

    if w == 2:  # Draw the figure of the function if could, time-consuming(which can be commented out).
        paint3d(func, limit)
        pass

    # Main code
    for i in range(cnt_num):
        start_time = time.time()

        oust_num = floor(random.random() * (n / (2 * up_rate) - n / up_rate) + n / up_rate)
        POS = limit * (-2 * np.random.rand(n, w) + 1)  # Set the initial Wolf pack position matrix, randomly generated

        # Primary iterative process
        for k in range(itr_num):
            Y = simulate_fc(POS, func)  # Calculate the initial odor concentration
            wolf_temp = wolf_sort(POS, Y)  # Sort the wolf pack by Y
            pos_head = wolf_temp[0, :w].reshape(1, -1)
            y_head = wolf_temp[0][wolf_temp.shape[1] - 1]
            if abs((-y_head) - extrenum) < gate:  # Succecced when it is close enough to the theoretical solution
                suc_cnt = suc_cnt + 1
                total_time = total_time + process_time
                RESULT.append(-y_head)
                print('No.' + str(suc_cnt) + '. ' + str(-y_head))
                break
            POS_EXP = wolf_temp[1:S_num + 1, :w]  #  Determine explore-wolfs
            Y_EXP = wolf_temp[1:S_num + 1, w]
            POS_FER = wolf_temp[S_num + 1:, :w]  # Determine fierce-wolfs
            Y_FER = wolf_temp[S_num + 1:, w]
            [POS_EXP, Y_EXP, pos_head, y_head] = BV_scout(POS_EXP, Y_EXP, pos_head, y_head, step_sc, sc_max, dir_num,
                                                          func)
            [POS_FER, Y_FER, pos_head, y_head] = BV_summon(POS_FER, Y_FER, pos_head, y_head, step_rn, step_bl, dist_min,
                                                           func)
            POS = np.vstack((np.vstack((pos_head, POS_EXP)), POS_FER))  # Merge
            POS = BV_update(POS, pos_head, oust_num, func)

        end_time = time.time()
        process_time = end_time - start_time

    print("The success rate is " + str(suc_cnt / cnt_num * 100) + "%.")
    print("The average search time is " + str(format(total_time / cnt_num, '.5f')) + " seconds.")
    ERROR = np.array(RESULT) - extrenum
    ERROR_ABS = np.abs(ERROR)
    max_index = np.argmax(ERROR_ABS)  # Find the worst element
    min_index = np.argmin(ERROR_ABS)  # Find the best element
    print("The best result is " + str(format(RESULT[min_index], '.5e')) + ".")
    print("The worst result is " + str(format(RESULT[max_index], '.5e')) + ".")
    print("The mean of the whole result is " + str(format(np.mean(RESULT), '.5e')) + ".")
    print("The standard division of the whole result is " + str(format(np.std(RESULT), '.5e')) + ".")
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(1):
        f_name = 'F2'
        WPA_MAIN(f_name)
