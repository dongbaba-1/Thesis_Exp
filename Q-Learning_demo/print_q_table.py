import numpy as np

# 加载 .npy 文件中的数据
q_table = np.load("Q_table.npy")

# 打印 Q-table 的形状
print("Q-table shape:", q_table.shape)

# 打印 Q-table 的内容
print("Q-table contents:")
print(q_table)

# # 打印 Q-table 的某些行或列
# print("Q-table first row:", q_table[0])
# print("Q-table first column:", q_table[:, 0])
