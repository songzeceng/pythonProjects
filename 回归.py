from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import pandas as pd
import numpy as np


def linearRegression():
    """
    线性回归预测波士顿房价
    :return:
    """
    # 获取数据
    lb = load_boston()

    # 分割数据(训练集和测试集)
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    print("测试集目标值：\n", y_test)
    print("训练集目标值：\n", y_train)

    # 标准化，避免误差
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train.reshape(-1, 1))  # 一维重构成二维
    x_test = std_x.transform(x_test.reshape(-1, 1))

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 预测
    # 正规方程求解预测
    lr = LinearRegression()
    lr.fit(x_train[:y_train.shape[0]], y_train)
    print("权重参数：", lr.coef_)
    y_predict = lr.predict(x_test)
    print("预测结果：", y_predict)
    print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict[:y_test.shape[0]]))

    print("*" * 200)

    # 梯度下降预测
    sr = SGDRegressor()
    sr.fit(x_train[:y_train.shape[0]], y_train)
    print("权重参数：", sr.coef_)
    y_predict = sr.predict(x_test)
    print("预测结果：", y_predict)
    print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict[:y_test.shape[0]]))

    print("*" * 200)

    # 岭回归
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train[:y_train.shape[0]], y_train)
    print("权重参数：", ridge.coef_)
    y_predict = ridge.predict(x_test)
    print("预测结果：", y_predict)
    print("均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict[:y_test.shape[0]]))


def logistic():
    """
    逻辑回归预测癌症性质
    多用于处理二分类问题
    :return:
    """
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses', 'Class']
    raw_data = pd.read_csv("D:\\develop\\PythonProjects\\MachineLearn\\breast-cancer-wisconsin.data",
                           names=column_names)

    print(raw_data)
    raw_data.replace(to_replace="?", value=np.nan, inplace=True)
    raw_data = raw_data.dropna()

    x_train, x_test, y_train, y_test = train_test_split(raw_data[column_names[1: 10]]
                                                        , raw_data[column_names[10]]
                                                        , test_size=0.25)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)
    # 目标值只有良性恶性两种情况，因此目标值不用标准化

    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    y_predict = lg.predict(x_test)
    print(lg.coef_)
    print("准确率：", lg.score(x_test, y_test))
    print("召回率：", classification_report(y_test, y_predict, labels=[2, 4]
                                        , target_names=["良性", "恶性"]))


if __name__ == '__main__':
    # linearRegression()
    logistic()
