import random

import numpy as np

"""图定义
A-B-C-D-E-F-G
1-2-3-4-5-6-7
"""

nodes = {
    1: np.asarray([[2, 6, 7], [12, 16, 14]], np.float32).T,
    2: np.asarray([[1, 3, 6], [12, 10, 7]], np.float32).T,
    3: np.asarray([[2, 4, 5, 6], [10, 3, 5, 6]], np.float32).T,
    4: np.asarray([[3, 5], [3, 4]], np.float32).T,
    5: np.asarray([[3, 4, 6, 7], [5, 4, 2, 8]], np.float32).T,
    6: np.asarray([[1, 2, 3, 5, 7], [16, 7, 6, 2, 9]], np.float32).T,
    7: np.asarray([[1, 5, 6], [14, 8, 9]], np.float32).T
}


def roulette_wheel_selection(probabilities):
    """
    根据给定的概率分布执行轮盘赌选择。

    参数:
        probabilities (list): 每个个体的选择概率列表。

    返回:
        int: 被选中的个体的索引。
    """
    # 确保概率之和为 1
    assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1"
    
    # 生成一个随机数，范围在 [0, 1) 内
    pick = random.random()
    # 计算累积概率
    cumulative_probability = 0.0
    for index, item_prob in enumerate(probabilities):
        cumulative_probability += item_prob
        if pick <= cumulative_probability:
            return index
    
    # 如果由于浮点数运算误差导致未能返回索引，则返回最后一个索引
    return len(probabilities) - 1


class Dijkstra:
    def __init__(self, nodes_dict, start_node: int):
        """算法初始化
        S/U的第一列表示节点编号
        对于S，第二列表示从源节点到本节点已求出的最小距离，不再变更
        对于U，第二列表示从源节点到本节点暂得的最小距离，可能变更
        """
        self.nodes_dict = nodes_dict
        self.start_node = start_node
        self.S = np.asarray([[self.start_node, 0]], np.float32)
        self.U = np.zeros([len(self.nodes_dict) - 1, 2], np.float32)
        
        _ = list(self.nodes_dict.keys())
        _.remove(self.start_node)
        self.U[:, 0] = _
        
        node_start_near = self.nodes_dict[self.start_node]
        self.U[:, 1] = [dict(zip(node_start_near[:, 0], node_start_near[:, 1]))[node]
                        if node in node_start_near[:, 0]
                        else np.inf
                        for node in self.U[:, 0]]
        
        # 最优路径及暂时最优路径的初始化
        # self.path_option = {4: [4]}
        # self.path_temp = {3: [4, 3], 4: [4], 5: [4, 5]}
        self.path_option = {self.start_node: [self.start_node]}
        self.path_temp = {self.start_node: [self.start_node]}
        for node in node_start_near[:, 0]:
            self.path_temp[node] = [self.start_node] + [node]
    
    def dijkstra_func(self):
        
        """算法初始化
        S/U的第一列表示节点编号
        对于S，第二列表示从源节点到本节点已求出的最小距离，不再变更
        对于U，第二列表示从源节点到本节点暂得的最小距离，可能变更
        """
        
        while self.U.size != 0:
            dist_min = np.min(self.U[:, 1])
            row_idx = np.argmin(self.U[:, 1])
            node_min = self.U[row_idx, 0]
            self.S = np.concat((self.S, [[node_min, dist_min]]))
            self.U = np.delete(self.U, row_idx, 0)
            
            # 将最小距离的节点添加到最优路径集合
            self.path_option[node_min] = self.path_temp[node_min]
            
            # 遍历最小距离节点的邻节点，判断是否在U集合中更新邻节点的距离
            nearby_nodes = self.nodes_dict[node_min][:, 0]
            for n in range(len(nearby_nodes)):
                node_temp = nearby_nodes[n]  # 节点
                idx_temp = np.where(self.U[:, 0] == node_temp)  # 节点在U索引
                if node_temp in self.U[:, 0]:  # 判断节点是否在U集合中
                    if dist_min + self.nodes_dict[node_min][:, 1][n] < self.U[idx_temp, 1]:
                        self.U[idx_temp, 1] = dist_min + self.nodes_dict[node_min][:, 1][n]
                        
                        # 更新暂时最优路径
                        self.path_temp[node_temp] = self.path_option[node_min] + [node_temp]
        
        print(np.asarray(self.S, dtype = np.int8))
        print(np.asarray(self.path_option[4], dtype = np.int8))
