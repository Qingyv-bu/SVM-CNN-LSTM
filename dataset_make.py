import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt
import matplotlib
import random
import scipy.io as scio
from sklearn import preprocessing
import torch
import pandas as pd
import sklearn
from joblib import dump, load

def open_data(base_path, key_num):
    path = base_path + str(key_num) + ".mat"
    str1 = "X" + "%03d" % key_num + "_DE_time"
    data = scio.loadmat(path)
    return data[str1]

# 时间步长 1024 和 重叠率 -0.5 
# window = 1024  step = 512  
# 切割划分方法: 参考论文 《时频图结合深度神经网络的轴承智能故障诊断研究》

def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):
    """
        data:要切分的时间序列数据,可以是一个一维数组或列表。
        time_steps:切分的时间步长,表示每个样本包含的连续时间步数。
        lable: 表示切分数据对应 类别标签
        overlap_ratio:前后帧切分时的重叠率,取值范围为 0 到 1,表示重叠的比例。
    """
    stride = int(time_steps * (1 - overlap_ratio))  # 计算步幅
    samples = (len(data) - time_steps) // stride + 1  # 计算样本数
    # 用于存储生成的数据
    Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])  
    # 记录数据行数(量)
    index_count = 0 
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(lable)  # 对应哪一类
        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
    return Clasiffy_dataFrame

# 数据集的制作
def make_datasets(hp=[0, 1, 2], fault_diameter=[0.007, 0.014, 0.021], split_rate = [0.7,0.2,0.1]):
    '''
        参数:
        data_file_csv: 故障分类的数据集,csv格式,数据形状: 119808行  10列
        label_list: 故障分类标签
        split_rate: 训练集、验证集、测试集划分比例

        返回:
        train_set: 训练集数据
        val_set: 验证集数据
        test_set: 测试集数据
    '''
    base_path1 = r"./data/"
    base_path2 = r"./data/"
    columns_name = ['hp_0_normal','hp_0_fd_0.007_inner','hp_0_fd_0.007_ball','hp_0_fd_0.007_outer',
                    'hp_0_fd_0.014_inner','hp_0_fd_0.014_ball','hp_0_fd_0.014_outer',
                    'hp_0_fd_0.021_inner','hp_0_fd_0.021_ball','hp_0_fd_0.021_outer',
                    'hp_1_normal','hp_1_fd_0.007_inner','hp_1_fd_0.007_ball','hp_1_fd_0.007_outer',
                    'hp_1_fd_0.014_inner','hp_1_fd_0.014_ball','hp_1_fd_0.014_outer',
                    'hp_1_fd_0.021_inner','hp_1_fd_0.021_ball','hp_1_fd_0.021_outer',
                    'hp_2_normal','hp_2_fd_0.007_inner','hp_2_fd_0.007_ball','hp_2_fd_0.007_outer',
                    'hp_2_fd_0.014_inner','hp_2_fd_0.014_ball','hp_2_fd_0.014_outer',
                    'hp_2_fd_0.021_inner','hp_2_fd_0.021_ball','hp_2_fd_0.021_outer',]
    data_list = pd.DataFrame()
    num = 0
    # 1.读取数据
    for i in hp:
        data = open_data(base_path1, 97 + i)
        data_list[columns_name[num]] = (data.reshape(-1))[:110000] # 统一维度
        num += 1

        for j in fault_diameter:
            if j == 0.007:
                inner_num = 105
                ball_num = 118
                outer_num = 130
            elif j == 0.014:
                inner_num = 169
                ball_num = 185
                outer_num = 197
            else:
                inner_num = 209
                ball_num = 222
                outer_num = 234

            inner_data = open_data(base_path2, inner_num + i)
            data_list[columns_name[num]] = (inner_data.reshape(-1))[:110000]
            num += 1

            ball_data = open_data(base_path2, ball_num + i)
            data_list[columns_name[num]] = (ball_data.reshape(-1))[:110000]
            num += 1

            outer_data = open_data(base_path2, outer_num + i)
            data_list[columns_name[num]] = (outer_data.reshape(-1))[:110000]
            num += 1
    # 2.分割样本点
    time_steps = 1024  # 时间步长
    overlap_ratio = 0.5  # 重叠率
    # 用于存储生成的数据# 10个样本集合
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])  
    # 记录类别标签
    label = 0
    # 使用iteritems()方法遍历每一列
    for column_name, column_data in data_list.items():
        # 对数据集的每一维进行归一化
        # column_data = normalize(column_data)
        # 划分样本点  window = 512  overlap_ratio = 0.5  samples = 467 每个类有467个样本
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        label += 1 # 类别标签递增
        samples_data = pd.concat([samples_data, split_data])
        # 随机打乱样本点顺序 
        samples_data = sklearn.utils.shuffle(samples_data) # 设置随机种子 保证每次实验数据一致

    # 3.分割训练集-、验证集、测试集
    sample_len = len(samples_data) # 每一类样本数量
    train_len = int(sample_len*split_rate[0])  # 向下取整
    val_len = int(sample_len*split_rate[1]) 
    train_set = samples_data.iloc[0:train_len,:]   
    val_set = samples_data.iloc[train_len:train_len+val_len,:]   
    test_set = samples_data.iloc[train_len+val_len:sample_len,:]   
    return  train_set, val_set, test_set, samples_data

def make_data_labels(dataframe):
    '''
        参数 dataframe: 数据框
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    # 信号值
    x_data = dataframe.iloc[:,0:-1]
    # 标签值
    y_label = dataframe.iloc[:,-1]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64')) # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return x_data, y_label

# 分量预处理
def imf_make_unify(data, lables, imfs_unify):
    '''
        参数 data: 待分解数据
            lables: 待分解数据对应标签
            imfs_unify: EMD分解分量个数统一标准

        返回 emd_result: 分解之后的数据集
            y_label    : 分解之后的数据集对应标签
    '''
    samples = data.shape[0]
    signl_len = data.shape[1]
    # 把数据转为numpy
    data = np.array(data)
    lables = np.array(lables)
    # 构造三维矩阵
    emd_result = np.zeros((samples, imfs_unify, signl_len))  
    # 创建 EMD 对象
    emd = EMD()
    # 待删除样本 行数
    delete_list_no = []
    # 对数据进行EMD分解
    for i in range(samples):
        IMFs= emd(data[i])  # 假设使用pyemd.emd进行EMD分解
        # 正常标准分量数量
        if len(IMFs) == imfs_unify:
            emd_result[i] = IMFs
        # 分量数目多的 合并
        elif len(IMFs) > imfs_unify:
            data_front  = IMFs[:imfs_unify-1, :] # 前 imfs_unify 的几组
            data_latter = IMFs[imfs_unify:, :] # 超过 imfs_unify 的后几组
            data_latter = np.sum(data_latter, 0) # 沿ｙ轴方向求和
            # 垂直合并
            merged_data = np.vstack((data_front, data_latter))
            emd_result[i] = merged_data
         # 小于7的样本不处理，默认结果都是0
        else:
            delete_list_no.append(i)

    # 删除 分量小于 7的样本
    for no in delete_list_no:
        # 使用 np.delete() 函数删除第 no 个元素
        emd_result = np.delete(emd_result, no, axis=0)
        lables = np.delete(lables, no, axis=0)

    # 把numpy转为  tensor
    emd_result = torch.tensor(emd_result).float()
    y_label = torch.tensor(lables, dtype=torch.int64)  # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return emd_result, y_label

train_set, val_set, test_set, samples_data = make_datasets()
# 制作标签
train_xdata, train_ylabel = make_data_labels(train_set)
val_xdata, val_ylabel = make_data_labels(val_set)
test_xdata, test_ylabel = make_data_labels(test_set)
# EMD分解预处理  统一保存7个分量
train_xdata, train_ylabel = imf_make_unify(train_xdata, train_ylabel, 7)
val_xdata, val_ylabel = imf_make_unify(val_xdata, val_ylabel, 7)
test_xdata, test_ylabel = imf_make_unify(test_xdata, test_ylabel, 7)
# 保存数据
dump(train_xdata, 'BFD_trainX')
dump(val_xdata, 'BFD_valX')
dump(test_xdata, 'BFD_testX')
dump(train_ylabel, 'BFD_trainY')
dump(val_ylabel, 'BFD_valY')
dump(test_ylabel, 'BFD_testY')