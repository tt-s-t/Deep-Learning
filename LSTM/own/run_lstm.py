from lstm_model import LSTM
import numpy as np
import math

class Dataset(object):
    def __init__(self,txt_data, sequence_length):
        self.txt_len = len(txt_data) #文本长度
        vocab = list(set(txt_data)) #所有字符合集
        self.n_vocab = len(vocab) #字典长度
        self.sequence_length = sequence_length
        self.vocab_to_index = dict((c, i) for i, c in enumerate(vocab)) #词-索引字典
        self.index_to_vocab = dict((i, c) for i, c in enumerate(vocab)) #索引-词字典
        self.txt_index = [self.vocab_to_index[i] for i in txt_data] #输入文本的索引表示

    def one_hot(self,input):
        onehot_encoded = []
        for i in input:
            letter = [0 for _ in range(self.n_vocab)] 
            letter[i] = 1
            onehot_encoded.append(letter)
        onehot_encoded = np.array(onehot_encoded)
        return onehot_encoded
    
    def __getitem__(self, index):
        return (
            self.txt_index[index:index+self.sequence_length],
            self.txt_index[index+1:index+self.sequence_length+1]
        )

#输入的有规律的序列数据
#txt_data = "abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc abc"
txt_data = "abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz "

#config
max_epoch = 5000
sequence_length = 28
dataset = Dataset(txt_data,sequence_length)
batch_num = math.ceil(dataset.txt_len /sequence_length) #向上取整
hidden_size = 32
lr = 1e-3

model = LSTM(dataset.n_vocab,hidden_size)

#训练
for epoch in range(max_epoch):
    h_prev = np.zeros((1, hidden_size))
    c_prev = np.zeros((1, hidden_size))
    loss = 0
    for b in range(batch_num):
        (x,y) = dataset[b]
        input = dataset.one_hot(x)
        target = dataset.one_hot(y)
        ps = model.forward(input,h_prev,c_prev) #注意：每个batch的h都是从0初始化开始，batch与batch间的隐藏状态没有关系
        loss += model.backward(target,lr)
    print("epoch: ",epoch)
    print("loss: ",loss/batch_num)

#预测
input_txt = 'a'
input_onehot = dataset.one_hot([dataset.vocab_to_index[input_txt]])
next_len = 50 #预测后几个word
h_prev = np.zeros((1, hidden_size))
c_prev = np.zeros((1, hidden_size))
pre_vocab = ['a']
pre_vocab1 = model.pre(input_onehot,h_prev,c_prev,next_len,dataset.index_to_vocab)
pre_vocab = pre_vocab + pre_vocab1
print(''.join(pre_vocab))
