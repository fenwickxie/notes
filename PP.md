<style>@import url(../auto_number_title.css); </style>

# Pathing Planning(路径规划)
+ 基于搜索的路径规划
+ 基于采样的路径规划

## 基于搜索的路径规划

通过搜索图形结构来找到最短或最优的路径，其中 A* 是最为常用和经典的算法之一

+  BFS(Breadth-First Searching，广度优先搜索)
  + 优点：可找到最短路径；适用于无权图
  + 缺点：时间复杂度高；空间复杂度高

+ DFS(Depth-First Searching，深度优先搜索)
  + 优点：空间复杂度低
  + 可能会陷入死循环；不一定能找到最短路径
 
+ Best-First Searching(最佳优先搜索)
  + 优点：速度快；可以处理启发式信息
  + 缺点：可能会陷入局部最优解
 
+ Dijkstra’s
  + 优点：可以找到最短路径；适用于有权图
  + 缺点：时间复杂度高；不能处理负权边。

+ A*
  + 优点：速度快；可以处理启发式信息；可以找到最短路径
  + 缺点：可能会陷入局部最优解

+ Bidirectional A*
  + 优点：速度快；可以找到最短路径
  + 缺点：需要存储两个搜索树；可能会出现问题，例如搜索空间过大或搜索树生长过慢

+ Anytime Repairing A*
![image](https://ucc.alicdn.com/pic/developer-ecology/cdzfr5ewdwyaw_78a209e3332b421798a041a3f4be5654.gif?x-oss-process=image%2Fresize%2Cw_1400%2Fformat%2Cwebp)
  + 优点：可以在任何时候停止搜索并返回最佳路径；可以处理启发式信息
  + 缺点：可能会陷入局部最优解

+ Learning Real-time A* (LRTA*，实时学习 A*)
  + 优点：可以处理动态环境；可以处理启发式信息。
  + 缺点：需要进行实时计算，可能会导致性能问题

+ Real-time Adaptive A* (RTAA*，实时自适应 A*)
  + 优点：可以处理动态环境；可以处理启发式信息
  + 缺点：需要进行实时计算，可能会导致性能问题

+ Lifelong Planning A* (LPA*，终身规划 A*)
  + 优点：可以在不同的时间段进行搜索；可以处理启发式信息
  + 缺点：需要存储大量的搜索树

```python
class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtStarSmart:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.fig, self.ax = plt.subplots()
        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.beacons = []
        self.beacons_radius = 2
        self.direct_cost_old = np.inf
        self.obs_vertex = self.utils.get_obs_vertex()
        self.path = None

    def planning(self):
        n = 0
        b = 2
        InitPathFlag = False
        self.ReformObsVertex()

        for k in range(self.iter_max):
            if k % 200 == 0:
                print(k)

            if (k - n) % b == 0 and len(self.beacons) > 0:
                x_rand = self.Sample(self.beacons)
            else:
                x_rand = self.Sample()

            x_nearest = self.Nearest(self.V, x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if x_new and not self.utils.is_collision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new)
                self.V.append(x_new)

                if X_near:
                    # choose parent
                    cost_list = [self.Cost(x_near) + self.Line(x_near, x_new) for x_near in X_near]
                    x_new.parent = X_near[int(np.argmin(cost_list))]

                    # rewire
                    c_min = self.Cost(x_new)
                    for x_near in X_near:
                        c_near = self.Cost(x_near)
                        c_new = c_min + self.Line(x_new, x_near)
                        if c_new < c_near:
                            x_near.parent = x_new

                if not InitPathFlag and self.InitialPathFound(x_new):
                    InitPathFlag = True
                    n = k

                if InitPathFlag:
                    self.PathOptimization(x_new)
                if k % 5 == 0:
                    self.animation()

        self.path = self.ExtractPath()
        self.animation()
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
        plt.pause(0.01)
        plt.show()

    def PathOptimization(self, node):
        direct_cost_new = 0.0
        node_end = self.x_goal

        while node.parent:
            node_parent = node.parent
            if not self.utils.is_collision(node_parent, node_end):
                node_end.parent = node_parent
            else:
                direct_cost_new += self.Line(node, node_end)
                node_end = node

            node = node_parent

        if direct_cost_new < self.direct_cost_old:
            self.direct_cost_old = direct_cost_new
            self.UpdateBeacons()

    def UpdateBeacons(self):
        node = self.x_goal
        beacons = []

        while node.parent:
            near_vertex = [v for v in self.obs_vertex
                           if (node.x - v[0]) ** 2 + (node.y - v[1]) ** 2 < 9]
            if len(near_vertex) > 0:
                for v in near_vertex:
                    beacons.append(v)

            node = node.parent

        self.beacons = beacons

    def ReformObsVertex(self):
        obs_vertex = []

        for obs in self.obs_vertex:
            for vertex in obs:
                obs_vertex.append(vertex)

        self.obs_vertex = obs_vertex

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)
        node_new = Node((x_start.x + dist * math.cos(theta),
                         x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start

        return node_new

    def Near(self, nodelist, node):
        n = len(self.V) + 1
        r = 50 * math.sqrt((math.log(n) / n))

        dist_table = [(nd.x - node.x) ** 2 + (nd.y - node.y) ** 2 for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if dist_table[ind] <= r ** 2 and
                  not self.utils.is_collision(node, nodelist[ind])]

        return X_near

    def Sample(self, goal=None):
        if goal is None:
            delta = self.utils.delta
            goal_sample_rate = self.goal_sample_rate

            if np.random.random() > goal_sample_rate:
                return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                             np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

            return self.x_goal
        else:
            R = self.beacons_radius
            r = random.uniform(0, R)
            theta = random.uniform(0, 2 * math.pi)
            ind = random.randint(0, len(goal) - 1)

            return Node((goal[ind][0] + r * math.cos(theta),
                         goal[ind][1] + r * math.sin(theta)))

    def SampleFreeSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.x_goal
```

+ Dynamic A* (D*，动态 A*)
  + 优点：可以处理动态环境；可以处理启发式信息
  + 缺点：需要存储大量的搜索树。

+ D* Lite
  + 优点：可以处理动态环境；可以处理启发式信息；空间复杂度低
  + 缺点：可能会陷入局部最优解。

+ Anytime D*
  + 优点：可以在任何时候停止搜索并返回最佳路径；可以处理动态环境；可以处理启发式信息
  + 缺点：可能会陷入局部最优解。

## 基于采样的路径规划

适用于复杂环境中的路径规划，如机器人导航、无人驾驶和物流配送等领域

+ RRT 
+ RRT-Connect
+ Extended-RRT
+ Dynamic-RRT
+ RRT*
+ Informed RRT*
+ RRT* Smart
+ Anytime RRT*
+ Closed-Loop RRT*
+ Spline-RRT*
+ Fast Marching Trees (FMT*)
+ Batch Informed Trees (BIT*)
