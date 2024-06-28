# SVM-CNN-LSTM

PyEMD库安装方法：pip install EMD-signal，如果之前通过pip install PyEMD安装了，需要卸载重装

data文件夹中包含30种分类的数据集

bearing_fault_diagnosis_SVM.py为SVM训练文件

dataset_make.py为CNN-LSTM数据集制作文件

bearing_fault_diagnosis_CNN_LSTM.py为CNN-LSTM训练文件


由于dataset_make.py制作的数据集文件过大，没有上传，可通过运行该文件对data中的文件进行整合，生成BFD_trainX、BFD_trainY、BFD_valX、BFD_valY、BFD_testX、BFD_testY这六个数据文件，分别为
训练集X\Y，验证集X\Y，测试集X\Y，该过程需要较长时间。在生成这些文件后才可运行bearing_fault_diagnosis_CNN_LSTM.py进行模型训练

best_model.pt为bearing_fault_diagnosis_CNN_LSTM.py生成的对于验证集正确率最高模型文件
