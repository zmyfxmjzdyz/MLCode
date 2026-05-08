# -*- coding: utf-8 -*-
"""
分类任务相关指标

@File  : classification_metrics.py
@Author: JinMing
@Date  : 2026/5/8 20:02
"""

import matplotlib.pyplot as plt
import numpy as np

"""
    1.混淆矩阵(二分类)ConfusionMatrix
    2.准确率 Accuracy
    3.精确率 Precision（查准率）
    4.召回率 Recall（查全率）
    5.F1-Score（P+R 调和平均）
    6.ROC曲线
    7.AUC面积
"""


def confusion_matrix(y_true, y_pred, y_which_true):
    """
    计算混淆矩阵
    :param y_true:真实值
    :param y_pred:预测值
    :param y_which_true:正例是什么
    :return:字典{'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum(np.logical_and(y_true == y_which_true, y_pred == y_which_true))
    FN = np.sum(np.logical_and(y_true == y_which_true, y_pred != y_which_true))
    FP = np.sum(np.logical_and(y_true != y_which_true, y_pred == y_which_true))
    TN = np.sum(np.logical_and(y_true != y_which_true, y_pred != y_which_true))
    return {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}


def confusion_matrix_by_threshold(y_true, y_pred_proba, y_which_true, y_which_false=None, threshold=0.5,
                                  bigger_is_true=True):
    """
    按阈值计算混淆矩阵
    :param y_true:真实值(离散)
    :param y_pred_proba:预测值(连续)
    :param y_which_true:正例是什么
    :param y_which_false:负例是什么
    :param threshold:阈值
    :param bigger_is_true:阈值大于等于threshold时，预测为正例，否则预测为负例
    :return:字典{'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
    """
    if y_which_false is None:
        y_which_false = set(y_true) - {y_which_true}
        y_which_false = list(y_which_false)[0]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred_proba)
    if bigger_is_true:
        y_pred = np.where(y_pred >= threshold, y_which_true, y_which_false)
    else:
        y_pred = np.where(y_pred < threshold, y_which_true, y_which_false)

    return confusion_matrix(y_true, y_pred, y_which_true)


def _roc_data(y_true, y_pred_proba, y_which_true):
    """
    计算roc曲线的数据
    :param y_true:
    :param y_pred_proba:
    :param y_which_true:
    :return:tpr, fpr
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    # p概率
    p = np.unique(np.append(y_pred_proba, [0., 1.]))
    # tpr, fpr
    confusion_matrix_list = [confusion_matrix_by_threshold(y_true, y_pred_proba, y_which_true, threshold=i) for i in p]
    tpr = [(i['TP'] / (i['TP'] + i['FN'])) if i['TP'] != 0 else 0 for i in confusion_matrix_list]
    fpr = [(i['FP'] / (i['FP'] + i['TN'])) if i['FP'] != 0 else 0 for i in confusion_matrix_list]
    return tpr, fpr


def roc_draw(y_true, y_pred_proba, y_which_true):
    """
    绘制roc曲线
    :param y_true:
    :param y_pred_proba:
    :param y_which_true:
    :return:无
    """
    y, x = _roc_data(y_true, y_pred_proba, y_which_true)
    plt.scatter(x, y)
    plt.plot(x, y, color='red')
    # plt.xlim(-0.1, 1)
    # plt.ylim(0, 1.1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()


def auc(y_true, y_pred_proba, y_which_true):
    """
    计算auc面积
    :param y_true:
    :param y_pred_proba:
    :param y_which_true:
    :return:auc面积
    """
    # 获取tpr,fpr数据
    y, x = _roc_data(y_true, y_pred_proba, y_which_true)

    # 计算AUC
    auc = 0
    last_y = 0
    last_x = 0

    for y1, y2, x1, x2 in zip(y, y[1:], x, x[1:]):
        auc += (y1 + y2) / 2 * abs(x2 - x1)

    return auc


if __name__ == '__main__':
    y_test = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 1, 0]
    print(confusion_matrix(y_test, y_pred, 1))
    y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    print(confusion_matrix_by_threshold(y_test, y_pred, 1, threshold=0.))
