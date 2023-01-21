# -*- coding: utf-8 -*-
 
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import os
 
train_novel_path ='./三国演义.txt'
vocab_key_dict_path ='./Chinese_vocab.txt' #词典
model_save_path = "./novel_creat_model.pkl" #保存整个网络
model_save_path_pth = "./novel_creat.pth" #保存网络参数
save_pred_novel_path ="./pred_novel.txt" #保存模型写下的小说
pred_novel_start_text='《残次品》'
 
class Dataset(torch.utils.data.Dataset):
    def __init__(self,args):
        self.args = args

        """加载数据集(小说)"""
        with open(train_novel_path,encoding='UTF-8') as f:
            self.words = f.read()

        """加载词典"""
        with open(vocab_key_dict_path, 'r',encoding='utf-8') as f:
            text=f.read()
        self.uniq_words = list(text)#转为列表形式，方便使用

        #将数据集转化为“词-索引”模式
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}  

        #将数据集转化为“索引-词”模式
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}      
        
        #把小说的 字 全部转换成 索引，这就是待会要进一步处理的数据
        self.words_indexes = []
        
        #把字典里没有的字符 用'*'代替表示
        for w in self.words:
            if w not in self.uniq_words:
                self.words_indexes.append(1482) #1482 =='*'
            else:
                self.words_indexes.append(self.word_to_index[w])
    
    #使得len(dataset)可以返回（文章长度-序列长度）
    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
 
    #使得可以作为一个迭代对象用[]访问（类似数组）,返回训练的x和y（一个预测下一个）
    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1])
        )
   
class GRUModel(nn.Module):
    def __init__(self, dataset):
        super(GRUModel, self).__init__()
        self.input_size = 128
        self.hidden_size = 256
        self.embedding_dim = self.input_size
        self.num_layers = 2

        n_vocab = len(dataset.uniq_words)#获取词典长度
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim #一个词用多少长度的向量来表示
        )

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.fc = nn.Linear(self.hidden_size, n_vocab) 
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output,state = self.gru(embed, prev_state)
        logits = self.fc(output)
 
        return logits,state
 
#模型训练   
def train(dataset, model, args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loader = DataLoader(dataset,batch_size=args.batch_size)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state = torch.zeros(model.num_layers, args.sequence_length, model.hidden_size)
        epoch_loss = 0
        for _, (x, y) in enumerate(train_loader):
            y_pred, state = model(x.to(device), state.to(device))
            
            loss = criterion(y_pred.transpose(1, 2), y.to(device)) #y_pred.transpose(1, 2)第一二维倒置
            state = state.detach() #返回一个新的tensor，从当前计算图中分离下来。但是仍指向原变量的存放位置，不同之处只是requirse_grad为false.得到的这个tensir永远不需要计算梯度，不具有grad.这样我们的state就不会在更新参数的时候被改变了。

            optimizer.zero_grad() #清空当前梯度
            loss.backward() #反向传播
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)#使用梯度裁剪
            optimizer.step() #参数更新
            epoch_loss += loss / len(train_loader)#累计loss
 
        print(f'EPOCH:{epoch:2}, train loss:{epoch_loss:.4f}')
    #模型保存
    torch.save(model, model_save_path)
    torch.save(model.state_dict(), model_save_path_pth)

def GetIndex(dict,word): #根据对应的词返回索引
    if word in dict:
        return dict[word]
    else:
        return dict['*']

#模型预测
def predict(dataset, model, text, next_words_len, args):
    for j in range(1,5):#写四章小说，每章next_words_len字
        if j != 1:
            text = content
        text += '第'+str(j)+'章'
        words = list(text) #将准备的文本转为列表
        words = words[args.sequence_length*(-1):]
        print(words)
        model.eval()  
        device = 'cpu'
        model.to(device)
        state = torch.zeros(model.num_layers, len(words), model.hidden_size)
        for i in range(0, next_words_len): #循环生成词汇
            x = torch.tensor([[GetIndex(dataset.word_to_index,w) for w in words[i:]]])#输入处理
            y_pred, state = model(x, state)
            last_word_logits = y_pred[0][-1] #得到预测的词的所属类别得分
            #word_index = torch.argmax(last_word_logits).item()#找出分数最大的索引值，并转为int
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(dataset.index_to_word[word_index])
        content =  "".join(words)
        with open(save_pred_novel_path, 'a+',encoding='utf-8') as wf:
            wf.write(content)
            wf.write('\n')

def main(args):
    dataset = Dataset(args)
    if os.path.exists(model_save_path): #如果有保存的网络，就根据保存的进度接着训练下去
        model = torch.load(model_save_path)
        print('发现有保存的Model,load model ....\n------开始训练----------')
    else: #如果没有则重头开始训练
        print('没发现有保存的Model,Creat model .... \n------开始训练----------')
        model = GRUModel(dataset)
    train(dataset, model, args)
    print("训练完成,开始生成文本")
    predict(dataset, model, pred_novel_start_text, 200, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gru')
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sequence-length', type=int, default=30) #每次训练多长的句子
    args = parser.parse_args([])
    main(args)