import numpy as np

nodes={
  1:np.asarray([[2,6,7],[12,16,14]],np.float32).T
  2:np.asarray([[1,3,6],[12,10,7]],np.float32).T
  3:np.asarray([[2,4,5,6],[10,3,5,6]],np.float32).T
  4:np.asarray([3,5],[3,4]],np.float32).T
  5:np.asarray([[3,4,6,7],[5,4,2,8]],np.float32).T
  6:np.asarray([[1,2,3,5,7],[16,7,6,2,9]],np.float32).T
  7:np.asarray([[1,5,6],[14,8,9]],np.float32).T
}

class Dijkstra:
  def __init__(self,nodes_dict,start_node:int):
    self.nodes_dict=nodes_dict
    self.S=np.asarray([[start_node,0]],np.float32)
    self.U=np.zeros([]len(slef.nodes_dict)-1,2],np.float32)
    self.U[:,0]=list(self.nodes_dict.keys()).remove(start_node)

    node_start_near=self.nodes_dict[start_node]
    self.U[:,1]=[dict(zip(node_start_near[:,0],node_start_near[:,1]))[node] if node in node_start_near[:,0] else np.inf for node in self.U[:,0]]

    # 最优路径及暂时最优路径的初始化
    self.path_option = {start_node:[start_node]}
    self.path_temp = {start_node:[start_node]}
    for node in node_start_near[:,0]:
      self.path_temp[node] = [start_node]+[node]
