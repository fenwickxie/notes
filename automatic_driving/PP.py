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
class ACO:
    def __init__(self, nodes_dict, start_node, end_node, num_ants: int, num_iter: int, eva_rate: float, alpha, beta, Q):
        # 始末节点
        self.start_node = start_node
        self.end_node = end_node
        # 蚁群相关定义
        self.num_ants = num_ants  # 蚂蚁数量
        self.eva_rate = eva_rate  # 信息素挥发因子，(0,1)区间
        self.alpha = alpha  # 信息素重要程度因子
        self.beta = beta  # 启发函数重要程度因子
        self.Q = Q  # 释放信息素常量
        # 迭代相关参数初始化
        self.num_iter = num_iter  # 循环次数，蚁群代数
        self.path_best = []  # 各代最佳路径
        self.length_best = []  # 各代最佳路径的长度
        self.length_avg = []  # 各代路径的平均长度
        
        # 将信息素、启发信息一并放入nodes_dict中
        for node in nodes_dict:
            pheromone_array = np.ones((nodes_dict[node].shape[0], 1), np.float32)
            nodes_dict[node] = np.hstack((nodes_dict[node], pheromone_array))
            nodes_dict[node] = np.hstack((nodes_dict[node], 1 / nodes_dict[node][:, 1:2]))
        
        self.nodes_dict = nodes_dict
def aco_func(self):
        # 蚁群代数循环
        for iteration in range(self.num_iter):
            path_ants = []
            length = []
            # 每代蚁群中每个蚂蚁循环
            for ant in range(self.num_ants):
                path_ant = []
                nodes_neighbor = []
                node_current = self.start_node
                path_ant.append(node_current)
                distance = 0.
                
                while self.end_node not in path_ant:  # 当终结点在路径列表时跳出循环
                    nodes_neighbor_with_distance = self.nodes_dict[node_current]
                    # search nearby node
                    nodes_neighbor = nodes_neighbor_with_distance[:, 0:1]
                    
                    # print(nodes_neighbor)
                    
                    # delete nodes those were visited
                    nodes_neighbor = [node for node in nodes_neighbor if node not in path_ant]
                    
                    # Judge whether it has entered a dead end, if it has, back to start and restart searching
                    if len(nodes_neighbor) == 0:
                        # init again
                        nodes_neighbor = []
                        node_current = self.start_node
                        path_ant.append(node_current)
                        distance = 0.
                        continue
                    
                    # calculate prob of next node
                    prob = []
                    for idx, node in enumerate(nodes_neighbor_with_distance[:, 0:1]):
                        if node in nodes_neighbor:
                            prob.append(
                                (nodes_neighbor_with_distance[idx, 2:3] ** self.alpha) * (nodes_neighbor_with_distance[
                                                                                          idx, 3:4] ** self.beta)
                            )
                    prob = np.asarray(prob) / np.sum(prob)
                    
                    # roulette wheel selection
                    node_target = nodes_neighbor[roulette_wheel_selection(prob)]
                    
                    # update the pheromone after pick the next node
                    nodes_neighbor_with_distance[:, 3:4] = (1 - self.eva_rate) * nodes_neighbor_with_distance[:, 3:4]
                    
                    # calculate single step distance
                    for index, node in enumerate(nodes_neighbor_with_distance[:, 0:1]):
                        if node == node_target:
                            distance = distance + nodes_neighbor_with_distance[index, 1:2]
                            
                            # update the pheromone of the picked node
                            nodes_neighbor_with_distance[index, 3:4] = 1 + nodes_neighbor_with_distance[index, 3:4]
                    
                    # update next node and path_ant
                    node_current = int(node_target)
                    path_ant.append(node_current)
                    
                # store the distance and path_ant of ant_i
                length.append(distance)
                path_ants.append(path_ant)
            
            # update path_best and length_best
            length_ndarray = np.asarray(length)
            min_index = np.argmin(length_ndarray)
            min_length = length[min_index]
            if iteration == 0:
                self.length_best = min_length
                self.length_avg = np.average(length_ndarray)
                self.path_best = path_ants[min_index]
            else:
                if min_length < self.length_best:
                    self.length_best = min_length
                    self.length_avg = np.average(length_ndarray)
                    self.path_best = path_ants[min_index]
            # if iteration == 0:
            #     self.length_best.append(min_length)
            #     self.length_avg.append(np.average(length_ndarray))
            #     self.path_best.append(path_ants[min_index])
            # else:
            #     if min_length < self.length_best:
            #         self.length_best.append(min_length)
            #         self.length_avg.append(np.average(length_ndarray))
            #         self.path_best.append(path_ants[min_index])
            #     else:
            #         self.length_best.append(self.length_best[-1])
            #         self.length_avg.append(self.length_avg.append[-1])
            #         self.path_best.append(self.path_best[-1])
        
        return self.length_best, self.path_best, self.length_avg


if __name__ == '__main__':
    dijkstra = Dijkstra(nodes, 1)
    dijkstra.dijkstra_func()
    
    aco = ACO(nodes, 1, 4, 50, 1000, 0.3, 1., 6., 1.)
    length_best, path_best, length_avg = aco.aco_func()
    print(path_best)
