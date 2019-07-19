from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def iris():
    ii = load_iris()


def news():
    '''
    朴素贝叶斯算法：假设各特征独立，准确率较高且稳定
    :return:
    '''
    news = fetch_20newsgroups(subset="all")  # 读取20则新闻
    print(news.data)
    print(news.target)
    print(news.DESCR)

    print("*"*20)

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 参数列表：特征值和目标值，测试集占比(25%的样本用来测试)
    # 返回值：训练集和测试集的特征值，训练集和测试集的目标值

    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    # 以训练集的特征值进行重要性分析，读取词汇等信息，转换成特征矩阵

    print(tf.get_feature_names())

    x_test = tf.transform(x_test)
    # 转换成测试词汇特征矩阵

    mult = MultinomialNB(alpha=1.0)  # 朴素贝叶斯算法
    mult.fit(x_train, y_train)  # 根据训练集的特征值和目标值进行朴素贝叶斯的学习
    y_predict = mult.predict(x_test)  # 根据由测试集的特征值，预测结果
    print("预测结果：", y_predict)
    print("准确率：", mult.score(x_test, y_test))
    # 由测试集的特征值和目标值，进行预测评估(函数里会调用predict函数进行预测)

    print("精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))


def decision():
    titan = pd.read_csv("D:\\develop\\PythonProjects\\MachineLearn\\titan.csv")
    # 读取数据

    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    # 处理数据，找出特征值和目标值

    x['age'].fillna(x['age'].mean(), inplace=True)

    print(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 分割数据，训练集和样本集

    dic = DictVectorizer(sparse=False)
    x_train = dic.fit_transform(x_train.to_dict(orient="records"))
    x_test = dic.fit_transform(x_test.to_dict(orient="records"))
    # 特征值转换成列表的列表

    print(x_train)
    print("-"*20)
    print(x_test)
    print("-" * 20)

    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)
    # 决策树学习训练集
    print("特征名：", dic.get_feature_names())

    print("准确度：", dec.score(x_test, y_test))
    # 评估决策树准确度
    export_graphviz(dec, "./output.dot", feature_names=["age", "pclass=1st", "pclass=2nd",
                                                        "pclass=3rd", "male", "female"])
    # 保存决策树，给出特征名称(尽量覆盖所有特征情况)

    print("****"*20)
    # 随机森林算法，最好的预测算法
    forest = RandomForestClassifier()
    gc = GridSearchCV(forest, param_grid={"n_estimators": [500, 800, 1200],
                          "max_depth": [5, 8, 15]}, cv=2)
    # 网格搜索，为算法寻找最好的参数模型，指定为2维搜索
    gc.fit(x_train, y_train)  # 网格搜索的训练
    print("准确度：", gc.score(x_test, y_test))  # 网格搜索的最高得分
    print("最优参数模型：", gc.best_params_)  # 最佳参数模型



if __name__ == '__main__':
    # iris()
    # news()
    decision()