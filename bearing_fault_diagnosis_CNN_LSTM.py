import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchinfo import summary

# 加载数据集
def dataloader(batch_size, workers=0):
    # 训练集
    train_xdata = load('BFD_trainX')
    train_ylabel = load('BFD_trainY')
    # 验证集
    val_xdata = load('BFD_valX')
    val_ylabel = load('BFD_valY')
    # 测试集
    test_xdata = load('BFD_testX')
    test_ylabel = load('BFD_testY')

    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                   batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                 batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
                                  batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    return train_loader, val_loader, test_loader

# 定义 EMDCNNLSTM 分类模型
class EMDCNNLSTMclassifier(nn.Module):
    def __init__(self, batch_size, input_dim, conv_archs, hidden_layer_sizes, output_dim, dropout_rate=0.5):
        super().__init__()
        # 批次量大小
        self.batch_size = batch_size
        # CNN参数
        self.conv_arch = conv_archs # cnn网络结构
        self.input_channels = input_dim # 输入通道数
        self.cnn_features = self.make_layers()

        # lstm层数
        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()  # 用于保存LSTM层的列表

        # 定义第一层LSTM   
        self.lstm_layers.append(nn.LSTM(conv_archs[-1][-1], hidden_layer_sizes[0], batch_first=True))
        # 定义后续的LSTM层
        for i in range(1, self.num_layers):
                self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))
                
        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )
    
    # CNN卷积池化结构
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def forward(self, input_seq): 
        # 数据预处理
        # 注意：这里是 把数据进行了堆叠 把一个7*1024 的矩阵 进行 划分堆叠成形状为 56 * 128， 就使输入序列的长度降下来了
        input_seq = input_seq.view(self.batch_size, -1, 128) 

        # CNN 卷积池化
        # CNN 网络输入[batch,H_in, seq_length]
        cnn_features = self.cnn_features(input_seq) # torch.Size([32, 256, 16])

        # 送入LSTM层
        # 改变输入形状，适应网络输入[batch, seq_length, H_in]
        lstm_out = torch.transpose(cnn_features, 1, 2)  # 反转维度 和序列长度 ，适应网络输入形状
        for lstm in self.lstm_layers:
            lstm_out, _= lstm(lstm_out)  ## 进行一次LSTM层的前向传播
        # print(lstm_out.size())  # torch.Size([32, 16, 128])
        out = self.classifier(lstm_out[:, -1, :]) # torch.Size([32, 10]  # 仅使用最后一个时间步的输出 
        return out

def model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader):
    model = model.to(device)
    # 样本长度
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size

    # 最高准确率  最佳模型
    best_accuracy = 0.0
    best_model = model

    train_loss = []     # 记录在训练集上每个epoch的loss的变化情况
    train_acc = []    # 记录在训练集上每个epoch的准确率的变化情况
    validate_acc = []
    validate_loss = []

    # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()

        loss_epoch = 0.    #保存当前epoch的loss和
        correct_epoch = 0  #保存当前epoch的正确个数和
        for seq, labels in train_loader: 
            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)  #   torch.Size([16, 10])
            # 对模型输出进行softmax操作，得到概率分布
            probabilities = F.softmax(y_pred, dim=1)
            # 得到预测的类别
            predicted_labels = torch.argmax(probabilities, dim=1)
            # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
            correct_epoch += (predicted_labels == labels).sum().item()
            # 损失计算
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
        # 计算准确率
        train_Accuracy  = correct_epoch/train_size 
        train_loss.append(loss_epoch/train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch+1:2} train_Loss: {loss_epoch/train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            loss_validate = 0.
            correct_validate = 0
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 对模型输出进行softmax操作，得到概率分布
                probabilities = F.softmax(pre, dim=1)
                # 得到预测的类别
                predicted_labels = torch.argmax(probabilities, dim=1)
                # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
                correct_validate += (predicted_labels == label).sum().item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()
            # print(f'validate_sum:{loss_validate},  validate_Acc:{correct_validate}')
            val_accuracy = correct_validate/val_size 
            print(f'Epoch: {epoch+1:2} val_Loss:{loss_validate/val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate/val_size)
            validate_acc.append(val_accuracy)
            # 如果当前模型的准确率优于之前的最佳准确率，则更新最佳模型
            #保存当前最优模型参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model# 更新最佳模型的参数

    # 保存最后的参数
    # torch.save(model, 'final_model_emd_cnn_lstm.pt')
    # 保存最好的参数
    torch.save(best_model, 'best_model.pt')
   
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    plt.plot(range(epochs), train_loss, color = 'b',label = 'train_loss')
    plt.plot(range(epochs), train_acc, color = 'g',label = 'train_acc')
    plt.plot(range(epochs), validate_loss, color = 'y',label = 'validate_loss')
    plt.plot(range(epochs), validate_acc, color = 'r',label = 'validate_acc')
    plt.legend()
    plt.show()   #显示 lable 
    print("best_accuracy :", best_accuracy)

if __name__=='__main__':
    # 参数与配置
    torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练
    batch_size = 256
    # 加载数据
    train_loader, val_loader, test_loader = dataloader(batch_size)
    # 定义模型参数
    input_dim = 7 * 8   # 输入维度为7个分量
    conv_archs = ((1, 64), (1, 128), (1, 256))   # CNN 层卷积池化结构  类似VGG
    hidden_layer_sizes = [256,128]
    output_dim = 30

    model = EMDCNNLSTMclassifier(batch_size, input_dim, conv_archs, hidden_layer_sizes, output_dim) 
    # summary(model, input_size=(256, 7, 1024))
    # 定义损失函数和优化函数 
    loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss
    learn_rate = 0.003
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器

    # 训练模型
    matplotlib.rc("font", family='Microsoft YaHei')
    epochs = 200
    # 模型训练
    model_train(batch_size, epochs, model, optimizer, loss_function, train_loader, val_loader)

    # 模型 测试集 验证  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练

    # 得出每一类的分类准确率
    model = torch.load('best_model.pt')
    model = model.to(device)

    # 使用测试集数据进行推断并计算每一类的分类准确率
    class_labels = []  # 存储类别标签
    predicted_labels = []  # 存储预测的标签

    with torch.no_grad():

        for test_data, test_label in test_loader:
            # 将模型设置为评估模式
            model.eval()
            test_data = test_data.to(device)
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
            class_labels.extend(test_label.tolist())
            predicted_labels.extend(predicted.tolist())

    # 混淆矩阵
    confusion_mat = confusion_matrix(class_labels, predicted_labels)

    from sklearn.metrics import classification_report
    # 计算每一类的分类准确率
    report = classification_report(class_labels, predicted_labels, digits=4)
    print(report)

    # 原始标签和自定义标签的映射
    label_mapping = {
        0: "C1",1: "C2",2: "C3",3: "C4",4: "C5",
        5: "C6",6: "C7",7: "C8",8: "C9",9: "C10",
        10: "C11",11: "C12",12: "C13",13: "C14",14: "C15",
        15: "C16",16: "C17",17: "C18",18: "C19",19: "C20",
        20: "C21",21: "C22",22: "C23",23: "C24",24: "C25",
        25: "C26",26: "C27",27: "C28",28: "C29",29: "C30",
    }

    # 绘制混淆矩阵
    plt.figure()
    sns.heatmap(confusion_mat,  xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    plt.show()