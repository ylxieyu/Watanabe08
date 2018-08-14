# 用来测试度量补偿方法
import Tools.DataSetTool as DataSetTool
import Tools.EvaluationTool as evaluationTool
import numpy as np
import sklearn.tree as tree


# 预测
# source_x, source_y 是训练集
# target_x, target_y 是预测列表
def predict(source_x, source_y, target_x, target_y, depth=40):
    # 训练模型
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=depth)
    clf.fit(source_x, source_y)
    # 预测
    for pre_index in range(len(target_x)):
        predictions = clf.predict(np.float32(DataSetTool.DataSetTool.metric_compensation
                                             (source_x, target_x[pre_index])))
        evaluationTool.EvaluationTool.get_output(predictions, target_y[pre_index])


# 17版预测
def predict_adopt(source_x, source_y, target_x, target_y, depth=40):
    # 训练模型
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=depth)
    # 预测
    for pre_index in range(len(target_x)):
        clf.fit(DataSetTool.DataSetTool.metric_compensation_adopt(source_x, target_x[pre_index]), source_y)
        predictions = clf.predict(target_x[pre_index])
        evaluationTool.EvaluationTool.get_output(predictions, target_y[pre_index])


# 迭代，核心
# x 实验用到的所有度量数据
# y 实验用到的所有标签数据
def main_iter(x, y, depth, index):
    # 每一次实验，弹出训练集，剩余的用作测试
    print('iter='+str(index+1))
    temp_x, temp_y = x, y
    t_x, t_y = x.copy(), y.copy()
    for iter_index in range(len(x)):
        source_x = t_x.pop(iter_index)
        source_y = t_y.pop(iter_index)
        target_x = t_x
        target_y = t_y
        predict(source_x, source_y, target_x, target_y, depth)
        t_x, t_y = temp_x.copy(), temp_y.copy()


# 17版
def main_iter_adopt(x, y, depth, index):
    # 每一次实验，弹出训练集，剩余的用作测试
    print('iter=' + str(index+1))
    temp_x, temp_y = x, y
    t_x, t_y = x.copy(), y.copy()
    for iter_index in range(len(x)):
        source_x = t_x.pop(iter_index)
        source_y = t_y.pop(iter_index)
        target_x = t_x
        target_y = t_y
        predict_adopt(source_x, source_y, target_x, target_y, depth)
        t_x, t_y = temp_x.copy(), temp_y.copy()


# 实验开始 begin
# 导入数据集
path = 'D:\\data\\'
x_list, y_list = DataSetTool.DataSetTool.init_data(path, 20)
# 训练100次
for i in range(100):
    main_iter(x_list, y_list, 35, i)
