import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


class ModifiedGridworld:
    def __init__(self, width=4, height=3, start_state=(0, 0), goal_state=(3, 2), bad_state=(3, 1), obstacle_state=(1, 1)):
        self.width = width
        self.height = height
        self.start_state = start_state
        self.goal_state = goal_state
        self.bad_state = bad_state
        self.obstacle_state = obstacle_state
        self.actions = ['上', '下', '左', '右']

    def move(self, state, action):
        if state == self.goal_state or state == self.bad_state:
            return state, self.get_reward(state)

        x, y = state
        reward = -0.04  # 默认奖励

        # 检查是否是边界或障碍物上的无效移动
        if (action == '上' and (y == self.height - 1 or (x, y + 1) == self.obstacle_state)) or \
                (action == '下' and (y == 0 or (x, y - 1) == self.obstacle_state)) or \
                (action == '左' and (x == 0 or (x - 1, y) == self.obstacle_state)) or \
                (action == '右' and (x == self.width - 1 or (x + 1, y) == self.obstacle_state)):
            # 给予负奖励并保持状态不变
            reward = -1
            new_state = state  # 保持状态不变
        else:
            # 处理有效移动
            if action == '上' and (x, y + 1) != self.obstacle_state:
                y += 1
            elif action == '下' and (x, y - 1) != self.obstacle_state:
                y -= 1
            elif action == '左' and (x - 1, y) != self.obstacle_state:
                x -= 1
            elif action == '右' and (x + 1, y) != self.obstacle_state:
                x += 1
            new_state = (x, y)
            reward = self.get_reward(new_state)

        return new_state, reward

    def get_reward(self, state):
        if state == self.goal_state:
            return 1
        elif state == self.bad_state:
            return -1
        else:
            return -0.04

    def is_terminal_state(self, state):
        return state == self.goal_state or state == self.bad_state


# 继承ModifiedGridworld类以实现TD学习
class TDGridworld(ModifiedGridworld):
    def __init__(self, width=4, height=3, start_state=(0, 0), goal_state=(3, 2), bad_state=(3, 1), obstacle_state=(1, 1), alpha=0.1, gamma=1.0):
        super().__init__(width, height, start_state, goal_state, bad_state, obstacle_state)
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.V = {(x, y): 0 for x in range(width) for y in range(height) if (x, y) != obstacle_state}  # 状态价值函数初始化
        self.Q = {((x, y), action): 0 for x in range(width) for y in range(height) for action in ['上', '下', '左', '右'] if (x, y) != obstacle_state}  # 行动价值函数初始化

    def update_value_function(self, state, next_state, reward):
        # 更新状态价值函数
        self.V[state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])

    def update_action_value_function(self, state, action, next_state, reward):
        # 更新行动价值函数
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * max(self.Q[(next_state, a)] for a in ['上', '下', '左', '右']) - self.Q[(state, action)])

    def simulate_td_learning(self, episodes=10000):
        for _ in range(episodes):
            state = self.start_state
            while not self.is_terminal_state(state):
                action = np.random.choice(self.actions)
                next_state, reward = self.move(state, action)
                self.update_value_function(state, next_state, reward)
                self.update_action_value_function(state, action, next_state, reward)
                state = next_state
        return self.V, self.Q

    def get_optimal_policy(self):
        optimal_policy = {}
        for state in self.V:
            if self.is_terminal_state(state):
                optimal_policy[state] = '结束'
            else:
                best_action = max(self.actions, key=lambda action: self.Q[(state, action)])
                optimal_policy[state] = best_action
        return optimal_policy


class LinearTDGridworld(TDGridworld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化线性模型的权重
        self.weights_v = np.zeros(self.width * self.height)
        self.weights_q = np.zeros((self.width * self.height, len(self.actions)))

    def state_to_features(self, state):
        # 将状态转换为特征向量（这里使用独热编码）
        x, y = state
        features = np.zeros(self.width * self.height)
        index = x * self.height + y
        features[index] = 1
        return features

    def update_linear_v(self, state, next_state, reward):
        # 更新状态价值函数的线性逼近模型
        features = self.state_to_features(state)
        next_features = self.state_to_features(next_state)
        prediction = np.dot(features, self.weights_v)
        next_prediction = np.dot(next_features, self.weights_v)
        td_error = reward + self.gamma * next_prediction - prediction
        self.weights_v += self.alpha * td_error * features

    def update_linear_q(self, state, action, next_state, reward):
        # 更新行动价值函数的线性逼近模型
        features = self.state_to_features(state)
        next_features = self.state_to_features(next_state)
        action_index = self.actions.index(action)
        prediction = np.dot(features, self.weights_q[:, action_index])
        next_predictions = np.dot(next_features, self.weights_q)
        td_error = reward + self.gamma * np.max(next_predictions) - prediction
        self.weights_q[:, action_index] += self.alpha * td_error * features


class MDPGridworld(ModifiedGridworld):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transition_probability(self, current_state, action, next_state):
        # 确定转移概率
        if self.is_terminal_state(current_state):
            return 1.0 if next_state == current_state else 0.0

        x, y = current_state
        intended_next_state = self.get_intended_next_state(current_state, action)

        # 如果下一个状态是预期的状态，返回主要概率
        if next_state == intended_next_state:
            return 0.8
        # 对于侧面的状态，返回较小的概率
        elif next_state in self.get_side_states(current_state, action):
            return 0.1
        # 如果状态没有改变（撞到墙或障碍物），返回剩余的概率
        elif next_state == current_state:
            return 0.1 if intended_next_state != current_state else 0.9
        return 0.0

    def get_intended_next_state(self, state, action):
        # 根据动作确定预期的下一个状态
        x, y = state
        if action == '上' and y < self.height - 1 and (x, y + 1) != self.obstacle_state:
            return x, y + 1
        elif action == '下' and y > 0 and (x, y - 1) != self.obstacle_state:
            return x, y - 1
        elif action == '左' and x > 0 and (x - 1, y) != self.obstacle_state:
            return x - 1, y
        elif action == '右' and x < self.width - 1 and (x + 1, y) != self.obstacle_state:
            return x + 1, y
        return state

    def get_side_states(self, state, action):
        # 根据动作确定侧面的状态
        x, y = state
        if action in ['上', '下']:
            side_states = [(max(x - 1, 0), y), (min(x + 1, self.width - 1), y)]
        else:
            side_states = [(x, max(y - 1, 0)), (x, min(y + 1, self.height - 1))]
        return [s for s in side_states if s != self.obstacle_state]

    def value_iteration(self, threshold=0.01, gamma=1.0):
        # 初始化价值函数
        V = {(x, y): 0 for x in range(self.width) for y in range(self.height) if (x, y) != self.obstacle_state}

        while True:
            delta = 0
            for state in V.keys():
                if self.is_terminal_state(state):
                    continue
                v = V[state]
                V[state] = max(
                    sum(self.transition_probability(state, action, next_state) *
                        (self.get_reward(next_state) + gamma * V[next_state])
                        for next_state in V.keys()) for action in self.actions
                )
                delta = max(delta, abs(v - V[state]))
            if delta < threshold:
                break
        return V

    def get_mdp_optimal_policy(self, V, gamma=1.0):
        # 根据价值函数获取最优策略
        policy = {}
        for state in V.keys():
            if self.is_terminal_state(state):
                policy[state] = '结束'
            else:
                best_action_value = float('-inf')
                best_action = None
                for action in self.actions:
                    action_value = sum(self.transition_probability(state, action, next_state) *
                                       (self.get_reward(next_state) + gamma * V[next_state])
                                       for next_state in V.keys())
                    if action_value > best_action_value:
                        best_action_value = action_value
                        best_action = action
                policy[state] = best_action
        return policy


mdp_env = MDPGridworld()
V = mdp_env.value_iteration()
optimal_policy = mdp_env.get_mdp_optimal_policy(V)

for state, action in optimal_policy.items():
    print(f"在状态 {state} 采取行动 '{action}'")
# 创建TD学习环境实例
td_env = TDGridworld()

# 执行TD学习
V, Q = td_env.simulate_td_learning()
optimal_policy = td_env.get_optimal_policy()
print(V)
print(Q)
for state, action in optimal_policy.items():
    print(f"在状态 {state} 采取行动 '{action}'")

# # 创建线性函数逼近环境实例
# linear_td_env = LinearTDGridworld()
#
# # 执行TD学习并获取线性逼近权重
# weights_v, weights_q = linear_td_env.simulate_td_learning()
# print(weights_v)
# print(weights_q)
#
# # 可视化状态价值函数（V）
# plt.figure(figsize=(6, 4))
# v_matrix = np.zeros((td_env.height, td_env.width))
# for state, value in V.items():
#     v_matrix[state[1], state[0]] = value
# plt.imshow(v_matrix, cmap='coolwarm', interpolation='nearest', extent=[0, td_env.width, 0, td_env.height], origin='lower')
# plt.colorbar(label='Value')
# plt.title('State Value Function (V)')
# plt.show()
#
# actions = ['上', '下', '左', '右']
# action_titles = ["Action Value Function (Q) - Action 'Up'",
#                  "Action Value Function (Q) - Action 'Down'",
#                  "Action Value Function (Q) - Action 'Left'",
#                  "Action Value Function (Q) - Action 'Right'"]
#
# # 创建细化的网格
# grid_x, grid_y = np.mgrid[0:td_env.width:100j, 0:td_env.height:100j]
#
# for i, action in enumerate(actions):
#     # 准备数据
#     state_coords = []
#     values = []
#     for ((x, y), act), value in Q.items():
#         if act == action:
#             state_coords.append((x, y))
#             values.append(value)
#
#     # 转换坐标列表为 NumPy 数组
#     state_coords = np.array(state_coords)
#     values = np.array(values)
#
#     # 插值
#     grid_z = griddata(state_coords, values, (grid_x, grid_y), method='cubic')
#
#     # 绘制三维图
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
#
#     # 添加颜色条
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     # 标题和标签
#     ax.set_title(action_titles[i])
#     ax.set_xlabel('State X-axis')
#     ax.set_ylabel('State Y-axis')
#     ax.set_zlabel('Value')
#
#     # 显示图形
#     plt.show()
#
# # 拟合函数可视化
# # 准备数据
# x_vals = np.array([key[0] for key in weights_v.keys()])
# y_vals = np.array([key[1] for key in weights_v.keys()])
# z_vals = np.array(list(weights_v.values()))
#
# # 创建细化的网格
# grid_x, grid_y = np.mgrid[0:td_env.width:100j, 0:td_env.height:100j]
#
# # 插值
# grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='cubic')
#
# # 绘制三维图
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制表面图
# surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
#
# # 添加颜色条
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# # 标题和标签
# ax.set_title('Approximated State Value Function')
# ax.set_xlabel('State X-axis')
# ax.set_ylabel('State Y-axis')
# ax.set_zlabel('Value')
#
# # 显示图形
# plt.show()