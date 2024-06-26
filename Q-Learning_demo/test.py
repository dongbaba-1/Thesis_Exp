import numpy as np
def discretize_state(x, target_x, bins=10):
    bin_edges = np.linspace(-10, 10, bins)
    state = np.digitize([x, abs(target_x - x)], bins=bin_edges)
    # 确保状态索引在合法范围内
   # state = np.clip(state, 0, bins - 1)
    return state

x = -11
target_x = 50
result = discretize_state(x, target_x)

print(result)  # 输出结果
