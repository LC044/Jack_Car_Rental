# -*- coding: utf-8 -*-
"""
@File    : calculate_value.py
@Author  : Shuaikang Zhou
@Time    : 2022/10/5 19:29
@IDE     : Pycharm
@Version : Python3.10
@comment : ···
"""
import numpy as np
from params import *

Tp = np.zeros(2 * 21 * 21).reshape(2, 21, 21)  # 状态转移概率矩阵，Tp[0,i,j]表示第一个租赁点状态从i到j的概率
reward = np.zeros(2 * 21).reshape(2, 21)  # 一步收益，2个地点，21个状态


def poisson(n, lamda):
    """
    计算泊松分布的概率
    :param lamda: λ
    :param n: 数量n
    :return:
    """
    return np.exp(-lamda) * (lamda ** n) / np.math.factorial(n)


def trans_prob(s, loc):
    """
    :param s: 初始车辆数
    :param loc: 租赁点位置 0:第一个租赁点 1:第二个租赁点
    :return:
    """
    for r in range(0, max_car_num + 1):  # 当天租出去的车数量，可以取到正无穷，但指数衰减取到MAX_CAR_GARAGE足够保证精度
        p_rent = poisson(r, request_mean[loc])  # 租出去r辆车的概率
        if p_rent < accurate:  # 精度限制
            return
        rent = min(s, r)  # 租车数不可能大于库存数
        reward[loc, s] += p_rent * rent_income * rent  # 租车收益
        for ret in range(0, max_car_num + 1):  # 当天还车数量ret
            p_ret = poisson(ret, return_mean[loc])  # 还ret辆车的概率
            if p_ret < accurate:  # 精度限制
                continue
            s_next = min(s - rent + ret, max_car_num)  # 下一步状态：租车+还车后的租车点汽车数量
            Tp[loc, s, s_next] += p_rent * p_ret  # 状态转移概率


def init_trans_prob():
    """
    计算状态转移概率
    """
    for i in range(0, max_car_num + 1):
        trans_prob(i, 0)
        trans_prob(i, 1)


def value_update(state, action, last_value):
    """
    更新当前状态的价值函数
    :param state: [i,j] i代表第一个租赁点的汽车数量，j代表第二个租赁点的汽车数量
    :param action: 动作
    :param last_value: 上一个价值函数
    :return: 当前状态的价值函数
    """
    if action > state[0]:
        action = state[0]  # 从租车点0移走的车数不可能大于库存
    elif action < 0 and -action > state[1]:
        action = -state[1]  # 从租车点1移走的车数不可能大于库存
    new_state = [state[0] - action, state[1] + action]
    new_state[0] = min(new_state[0], max_car_num)
    new_state[1] = min(new_state[1], max_car_num)
    # 移车之后状态从state变成new_state
    temp_v = -np.abs(action) * move_cost  # 移车代价
    for m in range(0, max_car_num + 1):
        for n in range(0, max_car_num + 1):  # 对所有后继状态(m,n)
            # temp_V 即是所求期望
            # Tp[0,i,j]表示第一个租赁点状态从i到j的概率
            # Tp[1,i,j]表示第二个租赁点状态从i到j的概率
            temp_v += Tp[0, new_state[0], m] * Tp[1, new_state[1], n] * (
                    reward[0, new_state[0]] + reward[1, new_state[1]] + discount * last_value[m, n])
    return temp_v
