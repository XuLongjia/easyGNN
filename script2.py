# # -*- coding: utf-8 -*-
# """
# -------------------------------------------------
#    File Name：     script2
#    Description :
#    Author :       Lr
#    date：          2023/1/11
# -------------------------------------------------
#    Change Activity:
#                    2023/1/11:
# -------------------------------------------------
# """
#
#
# # -*- coding: utf-8 -*-
# """
# -------------------------------------------------
#    File Name：     script
#    Description :
#    Author :       Liang Rong
#    date：          2023/1/4
# -------------------------------------------------
#    Change Activity:
#                    2023/1/4:
# -------------------------------------------------
# """
# import os
# import os.path as osp
# import pickle
# import numpy as np
# import itertools
# import scipy.sparse as sp
# import urllib
# from collections import namedtuple
#
#
#
# def read_data(path):
#     """使用不同的方式读取原始数据以进一步处理"""
#     name = osp.basename(path)
#     if name == "ind.cora.test.index":
#         out = np.genfromtxt(path, dtype="int64")
#         return out
#     else:
#         out = pickle.load(open(path, "rb"), encoding="latin1")
#         out = out.toarray() if hasattr(out, "toarray") else out
#         return out
#
# data_root="./data/cora_v2"
# filenames = ["ind.cora.{}".format(name) for name in
#              ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
#
#
# x, tx, allx, y, ty, ally, graph, test_index = [read_data(
#     osp.join(data_root, name)) for name in filenames]
#
#
# print("x = ", x.shape)
# print("tx = ", tx.shape)
# print("allx = ", allx.shape)
# print("y = ", y.shape)
# print("ty = ", ty.shape)
# print("ally = ", ally.shape)
# import pprint
#
# pprint.pprint(graph)
# print(len(graph.keys()))
#
#
# print("test_index = ", test_index.shape)
#
#
#
# from data import CoraData
#
# data = CoraData().data
#
#
#
# def sampling(src_nodes, sample_num, neighbor_table):
#     """根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
#     某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点
#
#     Arguments:
#         src_nodes {list, ndarray} -- 源节点列表
#         sample_num {int} -- 需要采样的节点数
#         neighbor_table {dict} -- 节点到其邻居节点的映射表
#
#     Returns:
#         np.ndarray -- 采样结果构成的列表
#     """
#     results = []
#     for sid in src_nodes:
#         if len(graph[sid]) >= sample_num:
#             res = np.random.choice(neighbor_table[sid], size=(sample_num,),replace=False)
#         else:
#             res = np.random.choice(neighbor_table[sid], size=(sample_num,),replace=True)
#
#         results.append(res)
#     return np.asarray(results).flatten()
#
#
# def multihop_sampling(src_nodes, sample_nums, neighbor_table):
#     sampling_result = [src_nodes]
#
#     for k, hopk_num in enumerate(sample_nums):
#         hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
#         sampling_result.append(hopk_result)
#     return sampling_result
#
#
# print("sampling = ",sampling([1],3,graph))
# print("sampling = ",sampling([2702],3,graph))
#
#
# print("multihop_sampling = ", multihop_sampling([1],[2,3],graph))
#
#
# for i in range(2,-1,-1):
#     print(f"fdasfas {i}")



# mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

import numpy as np

mx = np.array([[1,0,1],[0,1,0],[1,0,1]])
print(mx)

r_mat_inv_sqrt = np.array([[0.1,0,0],[0,0.2,0],[0,0,0.3]])
print(mx.dot(r_mat_inv_sqrt))


print("=====")
print(mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt))
print("======")

print(r_mat_inv_sqrt.dot(mx).dot(r_mat_inv_sqrt))