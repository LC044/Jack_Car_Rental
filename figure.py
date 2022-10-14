# -*- coding: utf-8 -*-
"""
@File    : figure.py
@Author  : Shuaikang Zhou
@Time    : 2022/10/5 20:41
@IDE     : Pycharm
@Version : Python3.10
@comment : ···
"""
import numpy as np
import matplotlib.pyplot as plt
from params import *


def draw_fig(value, policy, iteration):
    """
    绘图
    :param value: 价值函数
    :param policy: 策略
    :param iteration: 迭代次数
    :return:
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.matshow(policy, cmap=plt.cm.bwr, vmin=-max_move_num, vmax=max_move_num)
    ax.set_xticks(range(max_car_num + 1))
    ax.set_yticks(range(max_car_num + 1))
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.set_xlabel("Cars at second location")
    ax.set_ylabel("Cars at first location")
    for x in range(max_car_num + 1):
        for y in range(max_car_num + 1):
            ax.text(x=x, y=y, s=int(policy.T[x, y]), va='center', ha='center', fontsize=8)
    ax.set_title(r'$\pi_{}$'.format(iteration), fontsize=20)

    y, x = np.meshgrid(range(max_car_num + 1), range(max_car_num + 1))
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter3D(y, x, value.T)
    ax.set_xlim3d(0, max_car_num)
    ax.set_ylim3d(0, max_car_num)
    ax.set_xlabel("Cars at second location")
    ax.set_ylabel("Cars at first location")
    ax.set_title('value for ' + r'$\pi_{}$'.format(iteration), fontsize=20)
    plt.savefig(f'./result/{iteration}.png', bbox_inches='tight')
