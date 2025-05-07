from sklearn import datasets
from sklearn.model_selection import train_test_split
#逻辑回归线性模型
from sklearn.linear_model import LogisticRegression
#导入准确率预测

from sklearn.metrics import accuracy_score

#加载鸢尾花数据集

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(y)


#将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=77)

#创建一个逻辑回归对象,这里逻辑回归会根据我们的数据决定是用二分类还是多分类
# 逻辑回归到底是把多分类转行成了多个二分类还是用到的是softmax回归
#max_iter 迭代次数
lr = LogisticRegression(max_iter = 1000)

#lr = LogisticRegression(multi_class='ovr')# 多分类转行成了多个二分类
#lr = LogisticRegression(multi_class='multinomial')# softmax 回归做多分类


#使用训练集训练模型
lr.fit(X_train,y_train)

#对测试集进行预测
y_pred = lr.predict(X_test)

#打印模型的准确率
print("准确率： %.2f" % accuracy_score(y_test,y_pred))
