import numpy as np

class RNN(object):
    def __init__(self,input_size,hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(input_size, hidden_size)*0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
        self.b_h = np.zeros((1, hidden_size))
        self.W_hq = np.random.randn(hidden_size, input_size)*0.01
        self.b_q = np.zeros((1, input_size))
    
    def forward(self, inputs, h_prev): #targets是目标词的索引值(这样占的内存才会少)
        self.input = inputs
        #一次序列跑完后再更新参数
        self.hs, self.ps = {}, {} #字典形式存储
        self.hs[-1] = np.copy(h_prev) #隐藏变量赋予
        
        for t in range(len(inputs)):  
            self.hs[t] = np.tanh(np.matmul(inputs[t], self.W_xh) + np.matmul(self.hs[t-1], self.W_hh) + self.b_h) #隐藏状态 Ht. 
            ys = np.matmul(self.hs[t], self.W_hq) + self.b_q #输出
            self.ps[t] = np.exp(ys) / np.sum(np.exp(ys)) #实际输出（概率）——softmax
        return self.ps

    def backward(self, targets,lr):
        
        self.loss = 0 
        dWxh, dWhh, dWhq = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hq)
        dbh, dbq = np.zeros_like(self.b_h), np.zeros_like(self.b_q)
        dh = np.zeros_like(self.hs[0])

        T = len(self.input)
        for t in reversed(range(T)): #反过来开始，因为像隐藏状态求偏导那样，越往前面分支越多
            #loss计算
            label_onehot = np.zeros_like(self.ps[t])
            label_onehot[0, targets[t]] = 1.0#第几个样本最终属于哪一类(概率为1，其他为0)
            self.loss += -np.sum(np.log(self.ps[t]) * label_onehot)

            #梯度计算
            dy = (self.ps[t] - label_onehot)
            dWhq += np.matmul(self.hs[t].T,dy)
            dbq += dy 
            dh = np.matmul(np.matmul(np.linalg.matrix_power(self.W_hh.T,T-t),self.W_hq),dy.T).T + dh 
            dh_tanh = (1 - self.hs[t] * self.hs[t]) * dh # backprop through tanh nonlinearity #tanh'(x) = 1-tanh^2(x)
            dbh += dh_tanh
            dWxh += np.matmul(self.input[t].T.reshape(-1,1), dh_tanh)
            dWhh += np.matmul(dh_tanh, self.hs[t-1].T)
        
        #梯度裁剪(这里的限制范围需要自己根据需求调整，否则梯度太大会很难很难训练，loss会降不下去的)
        for dparam in [dWxh, dWhh, dWhq, dbh, dbq]: 
            np.clip(dparam, -0.5, 0.5, out=dparam)#限制在[-0.5,0.5]之间

        #参数更新
        self.W_xh += -lr * dWxh
        self.W_hh += -lr * dWhh
        self.W_hq += -lr * dWhq
        self.b_h += -lr * dbh
        self.b_q += -lr * dbq
        
        return self.loss

    def pre(self,input_onehot,h_prev,next_len,vocab): #input_onehot为输入的一个词的onehot编码，next_len为需要生成的单词长度，vocab是"索引-词"的词典
        xs, hs = {}, {} #字典形式存储
        hs[-1] = np.copy(h_prev) #隐藏变量赋予
        xs[0] = input_onehot
        pre_vocab = []
        for t in range(next_len):
            hs[t] = np.tanh(np.matmul(xs[t], self.W_xh) + np.matmul(hs[t-1], self.W_hh) + self.b_h) #隐藏状态 Ht. 
            ys = np.matmul(hs[t], self.W_hq) + self.b_q #输出
            ps = np.exp(ys) / np.sum(np.exp(ys))
            pre_vocab.append(vocab[np.argmax(ps)])
            #pre_vocab.append(vocab[np.random.choice(range(input_onehot.shape[1]), p=ps.ravel())])
            xs[t+1] = np.zeros((1, self.input_size)) # init
            xs[t+1][0,np.argmax(ps)] = 1
        return pre_vocab