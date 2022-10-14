# -*- coding: utf-8 -*-
"""
@File    : params.py
@Author  : Shuaikang Zhou
@Time    : 2022/10/5 20:39
@IDE     : Pycharm
@Version : Python3.10
@comment : ···
"""
# 初始化参数
max_car_num = 20  # 最大出租车个数
max_move_num = 5  # 最大移动车的数量
rent_income = 10  # 租车收益
request_mean = [3, 4]  # 租赁点租车请求期望
return_mean = [3, 2]  # 租赁点还车期望
discount = 0.9  # 折扣率
move_cost = 2  # 移动车的代价
accurate = 1e-6  # 泊松分布停止条件
