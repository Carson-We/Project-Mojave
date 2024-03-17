import matplotlib.pyplot as plt

# 機械人活動數據
robot_trajectory = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

# 創建一個圖形窗口
plt.figure()

# 提取X和Y坐標
x = [point[0] for point in robot_trajectory]
y = [point[1] for point in robot_trajectory]

# 繪製機械人活動軌跡
plt.plot(x, y, '-o', linewidth=2, markersize=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Movement Trajectory')
plt.grid(True)

# 顯示圖形
plt.show()
