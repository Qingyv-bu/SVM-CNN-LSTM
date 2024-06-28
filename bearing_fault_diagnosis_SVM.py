import scipy.stats
import scipy.io as scio
from sklearn import preprocessing
import numpy as np
import random
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from keras.utils import to_categorical


def open_data(base_path, key_num):
    path = base_path + str(key_num) + ".mat"
    str1 = "X" + "%03d" % key_num + "_DE_time"
    data = scio.loadmat(path)
    return data[str1]

def split_data(data, split_rate):
    length = len(data)
    num1 = int(length * split_rate[0])
    num2 = int(length * split_rate[1])
    index1 = random.sample(range(length), num1)
    train = data[index1]
    data = np.delete(data, index1, axis=0)
    index2 = random.sample(range(len(data)), num2)
    eval = data[index2]
    test = np.delete(data, index2, axis=0)
    return train, eval, test

def extract_time_domain_features(data):
    mean = np.mean(data, axis=1)  # 均值
    std_dev = np.std(data, axis=1)  # 标准差
    skewness = scipy.stats.skew(data, axis=1)  # 偏度
    kurtosis = scipy.stats.kurtosis(data, axis=1)  # 峰度
    return np.column_stack((mean, std_dev, skewness, kurtosis))

def extract_frequency_domain_features(data):
    fft_vals = np.fft.rfft(data, axis=1)  # 快速FFT
    fft_power = np.abs(fft_vals) ** 2  # 功率谱
    return fft_power

def deal_data_with_features(data, length, label):
    data = np.reshape(data, (-1))
    num = len(data) // length
    data = data[:num * length]
    data = np.reshape(data, (num, length))

    # Apply Min-Max Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(np.transpose(data, [1, 0]))
    data = np.transpose(data, [1, 0])

    # Extract Time Domain Features
    time_features = extract_time_domain_features(data)

    # Extract Frequency Domain Features
    frequency_features = extract_frequency_domain_features(data)

    # Combine all features
    combined_features = np.column_stack((time_features, frequency_features))

    # Add label
    label = np.ones((num, 1)) * label
    return np.column_stack((combined_features, label))

# 更新 load_data 函数以应用新的数据处理
def load_data_with_features(num=90, length=1280, hp=[0, 1, 2], fault_diameter=[0.007, 0.014, 0.021], split_rate=[0.7, 0.0, 0.3]):
    base_path1 = r"./data/"
    base_path2 = r"./data/"
    data_list = []
    label = 0
    for i in hp:
        normal_data = open_data(base_path1, 97 + i)
        data = deal_data_with_features(normal_data, length, label=label)
        label += 1
        data_list.append(data)

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
            inner_data = deal_data_with_features(inner_data, length, label)
            data_list.append(inner_data)
            label += 1

            ball_data = open_data(base_path2, ball_num + i)
            ball_data = deal_data_with_features(ball_data, length, label)
            data_list.append(ball_data)
            label += 1

            outer_data = open_data(base_path2, outer_num + i)
            outer_data = deal_data_with_features(outer_data, length, label)
            data_list.append(outer_data)
            label += 1

    num_list = [len(data) for data in data_list]
    min_num = min(num_list)
    min_num = min(num, min_num)

    train, eval, test = [], [], []
    for data in data_list:
        data = data[:min_num, :]
        a, b, c = split_data(data, split_rate)
        train.append(a)
        eval.append(b)
        test.append(c)

    train = np.vstack(train)
    train = train[random.sample(range(len(train)), len(train))]
    train_data = train[:, :-1]
    train_label = to_categorical(train[:, -1], 30)

    test = np.vstack(test)
    test = test[random.sample(range(len(test)), len(test))]
    test_data = test[:, :-1]
    test_label = to_categorical(test[:, -1], 30)

    # print(label)

    return train_data, train_label, test_data, test_label

# 重新加载数据并训练SVM
train_data, train_label, test_data, test_label = load_data_with_features()
print("训练数据形状:", train_data.shape)
print("训练标签形状:", train_label.shape)
print("测试数据形状:", test_data.shape)
print("测试标签形状:", test_label.shape)

# 优化范围
gamma_range = [1e-5, 1e1]
C_range = [1e-3, 1e3]

# 训练和评估SVM模型
svc = SVC()
model = make_pipeline(StandardScaler(), PCA(n_components=100, whiten=True), svc)
# print(train_label[-1])
model.fit(train_data, np.argmax(train_label, axis=1))

# eval_pred = model.predict(eval_data)
# eval_accuracy = np.mean(eval_pred == np.argmax(eval_label, axis=1))
# print("Evaluation Accuracy:", eval_accuracy)

test_pred = model.predict(test_data)
test_accuracy = np.mean(test_pred == np.argmax(test_label, axis=1))
print("Test Accuracy:", test_accuracy)

# 使用t-SNE进行降维
pca = PCA(n_components=100, whiten=True)
test_data_pca = pca.fit_transform(test_data)
# 绘制t-SNE结果
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制真实标签
scatter1 = ax.scatter(test_data_pca[:, 1], test_data_pca[:,70], c=np.argmax(test_label, axis=1), cmap=plt.get_cmap('tab20', len(np.unique(np.argmax(test_label, axis=1)))), alpha=0.6, s=50, label='True Classes')
legend1 = ax.legend(*scatter1.legend_elements(), title="True Classes", loc="upper left")
ax.add_artist(legend1)

# 绘制预测标签
scatter2 = ax.scatter(test_data_pca[:, 1], test_data_pca[:, 70], c=test_pred, cmap=plt.get_cmap('tab20', len(np.unique(test_pred))), marker='x', s=50, label='Predicted Classes')
legend2 = ax.legend(*scatter2.legend_elements(), title="Predicted Classes", loc="lower left")
ax.add_artist(legend2)

# 标注错误分类的点
incorrect_idx = np.where(np.argmax(test_label, axis=1) != test_pred)[0]
ax.scatter(test_data_pca[incorrect_idx, 0], test_data_pca[incorrect_idx, 1], facecolors='none', edgecolors='r', s=100, label='Misclassified')

ax.set_title('True Labels and Predicted Labels with Misclassifications')
ax.set_xlabel('t-SNE Feature 1')
ax.set_ylabel('t-SNE Feature 2')
ax.legend(loc='best')
plt.grid(True)
plt.show()

# # 绘制混淆矩阵
# mat = confusion_matrix(np.argmax(test_label, axis=1), test_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu")
# plt.xlabel('True Label')
# plt.ylabel('Predicted Label')
# plt.title('Confusion Matrix')
# plt.show()