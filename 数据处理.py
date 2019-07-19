from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy


def dicVec():
    """
    字典数据特征值化
    :return:
    """
    dic = DictVectorizer(sparse=False)
    data = dic.fit_transform([
        {"city": "anyang",
         "preference": 100},
        {"city": "xinxiang",
         "preference": 70},
        {"city": "luoyang",
         "preference": 80}
    ])
    print(dic.get_feature_names())
    print(data)


def countVec():
    """
    词频特征值化
    :return:
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["I love you", "Do you love me"])  # 统计单词，单个字母不统计
    print(cv.get_feature_names())
    print(data.toarray())


def chVec():
    """
    中文特征值化
    :return:
    """
    c1 = ' '.join(list(jieba.cut("让我将你心儿摘下，试着将它慢慢融化，看我在你心中是否仍完美无瑕")))
    c2 = ' '.join(list(jieba.cut("是否依然为我丝丝牵挂，依然爱我无法自拔，心中是否有我未曾到过的地方啊")))
    c3 = ' '.join(list(jieba.cut("那里湖面总是澄清，那里空气充满宁静")))
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])  # 统计单词，单个字母不统计
    print(cv.get_feature_names())
    print(data.toarray())


def tfidfVec():
    """
    tfidf特征值化
    :return:
    """
    c1 = ' '.join(list(jieba.cut("让我将你心儿摘下，试着将它慢慢融化，看我在你心中是否仍完美无瑕")))
    c2 = ' '.join(list(jieba.cut("是否依然为我丝丝牵挂，依然爱我无法自拔，心中是否有我未曾到过的地方啊")))
    c3 = ' '.join(list(jieba.cut("那里湖面总是澄清，那里空气充满宁静")))
    cv = TfidfVectorizer()
    data = cv.fit_transform([c1, c2, c3])  # 统计单词，单个字母不统计
    print(cv.get_feature_names())
    print(data.toarray())


def mm():
    """
    归一化处理，受异常点影响大
    适合精确度小的场景
    :return: None
    """
    mm = MinMaxScaler()
    data = mm.fit_transform([[12, 32, 54], [32, 67, 91], [15, 31, 63]])
    print(data)


def im():
    """
    缺失值处理
    :return:
    """
    im = Imputer(missing_values="NaN", strategy="mean", axis=0)  # 填充nan， 填充值为平均数，axis=0列
    print(im.fit_transform([[1, numpy.nan, 3], [numpy.nan, 0, numpy.nan], [12, 3, 9]]))


def standardScale():
    """
    标准化，受异常点影响小
    适合大数据场景
    :return:
    """
    ss = StandardScaler()
    data = ss.fit_transform([[-7, 32, 54], [32, 67, 91], [15, 31, 63]])
    print(data)


def val():
    """
    数据降维，去除相同的列或行
    :return:
    """
    va = VarianceThreshold(threshold=0.0)
    print(va.fit_transform([[0, 2, 3], [0, 1, 3], [0, 4, 3]]))


def pca():
    """
    主成分分析降维
    :return:
    """
    print(PCA(n_components=0.93).fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]))
    # 保留93%个主成分


if __name__ == '__main__':
    # dicVec()
    # countVec()
    # chVec()
    # tfidfVec()
    # mm()
    # standardScale()
    # im()
    # val()
    pca()
    print("----------")
