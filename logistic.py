# -*- coding: utf-8 -*-
import os
import math

import scipy.io as scio
import numpy as np
from sklearn.linear_model import LogisticRegression #加载模块
from sklearn.linear_model import LogisticRegressionCV


def LoadData(file_path):
    '''
    加载mat文件
    :param file_path: 文件路径
    :return:
    '''
    data = scio.loadmat(file_path)
    return data


def StndTrainData(x, axis):
    '''
    标准化训练数据，每列使它们满足零均值和单位方差
    :param x: numpy.ndarray类型
    :param axis:
    :return: 处理后的训练数据，均值，均方差
    '''
    #不要修改原数据
    x = np.array(x)   #拷贝
    xr = np.rollaxis(x, axis=axis)
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    xr -= mean #减去均值
    xr /= std #除以均方差
    return xr, mean, std


def StndTestData(test, mean, std):
    '''
    标准化测试数据
    :param test: 测试数据
    :param mean: 训练数据的均值
    :param std:  训练数据的均方差
    :return: 处理后的测试数据
    '''
    #不要修改原数据
    testr = np.array(test)
    testr -= mean #减去均值
    testr /= std #除以均方差
    return testr


def Log(x):
    '''
    使用log(x[i][j] + 0.1)变换特征数据
    :param x: 原数据
    :return: 处理后的数据
    '''
    #不要修改原数据
    xr = np.array(x) #copy
    xr = np.log(xr + 0.1)
    return xr


def Binary(x):
    '''
    二值化数据 >0为1 否则为0
    :param x: 原数据
    :return: 处理后的数据
    '''
    #不要修改原数据
    xr = np.array(x) #copy
    xr = np.where(xr>0, 1, 0)
    return xr


def Sigmod(z):
    '''
    for i in xrange(np.size(z)):
        try:
            a = math.exp(-z[i])
        except:
            pass
    '''
    return 1.0 / (1.0 + np.exp(-z))


def ChangeWeight(w1, w2):
    '''
    计算两次权重改变了多少
    :param w1:
    :param w2:
    :return:
    '''
    return math.sqrt(np.sum(np.square(w1 - w2)) / np.shape(w1)[0])


def GradAscent(x, y, l2_param=0.05, round=2000, step=0.001):
    '''
    梯度上升算法，采用L2正则化方法
    :param x: 数据集
    :param y: 结果集
    :return: 权重
    '''
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    m, n = np.shape(x)
    weights = np.mat(np.ones((n, 1)))
    #weights = np.matrix(np.random.rand(n) - 0.5).T  # 初始化系数
    #delta = np.mat(np.full((n, 1), 0))
    change = 1.0
    i = 0
    #print(l2_param, step, round)
    while change > 0.0001 and i < round:
    #while i < 500:
        w = np.mat(weights)
        z = Sigmod(x * weights)
        error = y - z # + delta
        weights = weights + step * x.T * error - step * l2_param * weights
        #delta = np.mat(np.full((n, 1), math.sqrt(np.sum(np.square(weights))/m)))
        change = ChangeWeight(weights, w)
        '''if change > 0.005:
            alpha = 0.0003
        else:
            alpha = 0.0001'''
        i += 1
    #print i, change
    return weights


def Classify(data, weights):
    '''
    对数据进行分类
    :param data:
    :param weights:
    :return:
    '''
    data = np.asmatrix(data)
    z = Sigmod(data * weights)
    for i in range(len(z)):
        if z[i][0] > 0.5:
            z[i][0] = 1.0
        else:
            z[i][0] = 0.0
    return z


def CaclErrorRate(test_ret, real_ret):
    '''
    计算错误率
    :param test_ret: 测试结果
    :param real_ret: 真实结果
    :return:
    '''
    count = len(test_ret)
    count_error = 0
    for i in range(count):
        if test_ret[i][0] != real_ret[i][0]:
            count_error += 1
    return count_error / float(count)


def CrossValidator(data):
    '''
    十折交叉验证
    :param data:
    :return:
    '''
    params = []
    err_set = []
    l2_param = [0.0, 0.0, 0.0]
    for k in xrange(20):
        params.append(1.0 + 1.0 * k)
        err_set.append([0.0, 0.0, 0.0])
    splits = np.array_split(data['Xtrain'], 5)
    splits_y = np.array_split(data['ytrain'], 5)
    train_set = np.ndarray([])
    validate_set = np.ndarray([])
    train_set_y = np.ndarray([])
    validate_set_y = np.ndarray([])
    is_train = False
    for i in xrange(5): #第i个切块是验证集
        print i
        for j in xrange(5):
            if j == i:
                validate_set = splits[i]
                validate_set_y = splits_y[i]
            else:
                if False == is_train:
                    train_set = splits[i]
                    train_set_y = splits_y[i]
                    is_train = True
                else:
                    train_set = np.concatenate((train_set, splits[j]))
                    train_set_y = np.concatenate((train_set_y, splits_y[j]))
        # 1、使用均值和单位方差进行标准化
        train_set_stnd, mean, std = StndTrainData(train_set, 0)
        validate_set_stnd = StndTestData(validate_set, mean, std)
        for k in xrange(20):
            weights = GradAscent(train_set_stnd, train_set_y, params[k], 2000)
            ret = Classify(train_set_stnd, weights)
            print(CaclErrorRate(ret, train_set_y))
            # 对验证数据分类，计算错误率
            ret = Classify(validate_set_stnd, weights)
            err_rate = CaclErrorRate(ret, validate_set_y)
            err_set[k][0] += err_rate
            print(params[k], err_rate, err_set[k][0])

        # 2、Log
        train_set_log = Log(train_set)
        validate_set_log = Log(validate_set)
        for k in xrange(20):
            weights = GradAscent(train_set_log, train_set_y, params[k], 2000)
            ret = Classify(train_set_log, weights)
            print(CaclErrorRate(ret, train_set_y))
            # 对验证数据分类，计算错误率
            ret = Classify(validate_set_log, weights)
            err_rate = CaclErrorRate(ret, validate_set_y)
            err_set[k][1] += err_rate
            print(params[k], err_rate, err_set[k][1])

        # 3、Binary
        train_set_bin = Binary(train_set)
        validate_set_bin = Binary(validate_set)
        for k in xrange(20):
            weights = GradAscent(train_set_bin, train_set_y, params[k], 2000)
            ret = Classify(train_set_bin, weights)
            print(CaclErrorRate(ret, train_set_y))
            # 对验证数据分类，计算错误率
            ret = Classify(validate_set_bin, weights)
            err_rate = CaclErrorRate(ret, validate_set_y)
            err_set[k][2] += err_rate
            print(params[k], err_rate, err_set[k][2])

    for i in xrange(3):
        min_err_index = 0
        for k in xrange(1, 20):
            if err_set[min_err_index][i] > err_set[k][i]:
                min_err_index = k
        l2_param[i] = params[min_err_index]

    return  l2_param


def main():
    #加载数据
    data = LoadData('spamData.mat')

    #交叉验证速度很慢，注释掉了，返回结果为[20.0, 10.0, 1.0]
    #l2_param = CrossValidator(data)
    l2_param = [20.0, 10.0, 1.0]

    print("%-8s\t%-8s\t%-8s") % ("method", "train", "test")

    #1、使用均值和单位方差进行标准化
    print("%-8s\t") % ("Stnd"),
    train_stnd, mean, std = StndTrainData(data['Xtrain'], 0)
    # 梯度上升法拟合回归参数
    weights = GradAscent(train_stnd, data['ytrain'], l2_param[0])
    #print(weights)
    # 对训练数据分类，计算错误率
    ret = Classify(train_stnd, weights)
    print("%8.6f\t") % CaclErrorRate(ret, data["ytrain"]),
    # 对测试数据分类，计算错误率
    test_stnd = StndTestData(data['Xtest'], mean, std)
    ret = Classify(test_stnd, weights)
    print("%8.6f\t") % (CaclErrorRate(ret, data["ytest"]))

    #2、Log
    print("%-8s\t") % ("Log"),
    train_log = Log(data['Xtrain'])
    # 梯度上升法拟合回归参数
    weights = GradAscent(train_log, data['ytrain'], l2_param[1])
    #print(weights)
    # 对训练数据分类，计算错误率
    ret = Classify(train_log, weights)
    print("%8.6f\t") % (CaclErrorRate(ret, data["ytrain"])),
    # 对测试数据分类，计算错误率
    test_log = Log(data['Xtest'])
    ret = Classify(test_log, weights)
    print("%8.6f\t") % (CaclErrorRate(ret, data["ytest"]))

    #3、Binary
    print("%-8s\t") % ("Binary"),
    train_bin = Binary(data['Xtrain'])
    # 梯度上升法拟合回归参数
    weights = GradAscent(train_bin, data['ytrain'], l2_param[2])
    #print(weights)
    # 对训练数据分类，计算错误率
    ret = Classify(train_bin, weights)
    print("%8.6f\t") % (CaclErrorRate(ret, data["ytrain"])),
    # 对测试数据分类，计算错误率
    test_bin = Binary(data['Xtest'])
    ret = Classify(test_bin, weights)
    print("%8.6f\t") % (CaclErrorRate(ret, data["ytest"]))

    '''
    #调用sklearn库测试 结果为0.07145187601957581  0.06901041666666663
    model = LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20, fit_intercept=True)
    #model = LogisticRegressionCV(cv=10, penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20, fit_intercept=True)
    model.fit(data['Xtrain'], data['ytrain'])  # 对模型进行训练
    acc = model.score(data['Xtrain'], data['ytrain'])  # 根据给定数据与标签返回正确率的均值
    print 1-acc #0.07145187601957581
    acc = model.score(data['Xtest'], data['ytest'])  # 根据给定数据与标签返回正确率的均值
    print 1-acc #0.06901041666666663
    '''


if __name__ == "__main__":
    main()