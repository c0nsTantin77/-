import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']           # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False             # 正常显示负号

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 加载糖尿病数据集
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 拆分为训练集和测试集，固定随机种子
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测训练集和测试集
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

# 打印均方误差
print("训练集均方误差：%.2f" % mean_squared_error(y_train, y_pred_train))
print("测试集均方误差：%.2f" % mean_squared_error(y_test, y_pred_test))

# ---------------------
# 可视化：预测值 vs 真实值（测试集）
# ---------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_test, color='blue', label='预测值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='理想对角线')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('线性回归预测效果（测试集）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
