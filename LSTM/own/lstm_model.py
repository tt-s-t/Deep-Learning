import numpy as np

def sigmoid(x):
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)

def tanh(x):
    result = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return result

class LSTM(object):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        #输入门
        self.Wxi = np.random.randn(input_size, hidden_size)
        self.Whi = np.random.randn(hidden_size, hidden_size)
        self.B_i  = np.zeros((1, hidden_size))
        #遗忘门
        self.Wxf = np.random.randn(input_size, hidden_size)
        self.Whf = np.random.randn(hidden_size, hidden_size)
        self.B_f = np.zeros((1, hidden_size))
        #输出门
        self.Wxo = np.random.randn(input_size, hidden_size)
        self.Who = np.random.randn(hidden_size, hidden_size)
        self.B_o = np.zeros((1, hidden_size))
        #候选记忆细胞
        self.Wxc = np.random.randn(input_size, hidden_size)
        self.Whc = np.random.randn(hidden_size, hidden_size)
        self.B_c = np.zeros((1, hidden_size))
        #输出
        self.W_hd = np.random.randn(hidden_size, input_size)
        self.B_d = np.zeros((1, input_size))

    def forward(self,X,Ht_1,Ct_1): #前向传播
        #存储
        self.it_stack = {} #输入门存储
        self.ft_stack = {} #遗忘门存储
        self.ot_stack = {} #输出门存储
        self.cc_stack = {} #候选记忆细胞存储
        self.c_stack = {} #记忆细胞存储
        self.X_stack = {} #X存储
        self.Ht_stack = {} #隐藏状态存储
        self.Y_stack = {} #输出存储

        self.Ht_stack[-1] = Ht_1
        self.c_stack[-1] = Ct_1
        self.T = X.shape[0]

        for t in range(self.T):
            self.X_stack[t] = X[t].reshape(-1,1).T
            #输入门
            net_i = np.matmul(self.X_stack[t], self.Wxi) + np.matmul(self.Ht_stack[t-1], self.Whi) + self.B_i
            it = sigmoid(net_i)
            self.it_stack[t] = it
            #遗忘门
            net_f = np.matmul(self.X_stack[t], self.Wxf) + np.matmul(self.Ht_stack[t-1], self.Whf) + self.B_f
            ft = sigmoid(net_f)
            self.ft_stack[t] = ft
            #输出门
            net_o = np.matmul(self.X_stack[t], self.Wxo) + np.matmul(self.Ht_stack[t-1], self.Who) + self.B_o
            ot = sigmoid(net_o)
            self.ot_stack[t] = ot
            #候选记忆细胞
            net_cc = np.matmul(self.X_stack[t], self.Wxc) + np.matmul(self.Ht_stack[t-1], self.Whc) + self.B_c
            cct = tanh(net_cc)
            self.cc_stack[t] = cct
            #记忆细胞
            Ct = ft*self.c_stack[t-1]+it*cct
            self.c_stack[t] = Ct
            #隐藏状态
            Ht = ot*tanh(Ct)
            self.Ht_stack[t] = Ht
            #输出
            y = np.matmul(Ht, self.W_hd) + self.B_d
            Yt = np.exp(y) / np.sum(np.exp(y)) #softmax
            self.Y_stack[t] = Yt

    def backward(self,target,lr):
        #初始化
        dH_1, dnet_ct_1 = np.zeros([1,self.hidden_size]), np.zeros([1,self.hidden_size])

        dWxi, dWhi, dBi = np.zeros_like(self.Wxi), np.zeros_like(self.Whi), np.zeros_like(self.B_i)
        dWxf, dWhf, dBf = np.zeros_like(self.Wxf), np.zeros_like(self.Whf), np.zeros_like(self.B_f)
        dWxo, dWho, dBo = np.zeros_like(self.Wxo), np.zeros_like(self.Who), np.zeros_like(self.B_o)
        dWxc, dWhc, dBc = np.zeros_like(self.Wxc), np.zeros_like(self.Whc), np.zeros_like(self.B_c)
        dWhd,dBd = np.zeros_like(self.W_hd),np.zeros_like(self.B_d)

        self.loss = 0

        for t in reversed(range(self.T)): #反过来开始，越往前面分支越多       
            dY = self.Y_stack[t] - target[t].reshape(-1,1).T
            self.loss += -np.sum(np.log(self.Y_stack[t]) * target[t].reshape(-1,1).T)
            #对输出的参数
            dWhd += np.matmul(self.Ht_stack[t].T,dY)
            dBd += dY

            dH = np.matmul(dY, self.W_hd.T) + dH_1 #dH更新

            #对有关输入门，遗忘门，输出门，候选记忆细胞中参数的求导的共同点
            temp = tanh(self.c_stack[t])
            dnet_ct = dH * self.ot_stack[t] * (1-temp*temp) + dnet_ct_1 #记忆细胞
            dnet_cct = dnet_ct * self.it_stack[t] * (1 - self.cc_stack[t]*self.cc_stack[t]) #候选记忆细胞
            dnet_o = dH * temp * self.ot_stack[t] * (1 - self.ot_stack[t]) #输出门
            dnet_f = dnet_ct * self.c_stack[t-1] * self.ft_stack[t] * (1 - self.ft_stack[t]) #遗忘门
            dnet_i = dnet_ct * self.cc_stack[t] * self.it_stack[t] * (1 - self.it_stack[t]) #输入门

            #候选记忆细胞中参数
            dWxc += np.matmul(self.X_stack[t].T, dnet_cct)
            dWhc += np.matmul(self.Ht_stack[t-1].T, dnet_cct)
            dBc += dnet_cct

            #输出门
            dWxo += np.matmul(self.X_stack[t].T, dnet_o)
            dWho += np.matmul(self.Ht_stack[t-1].T, dnet_o)
            dBo += dnet_o

            #遗忘门
            dWxf += np.matmul(self.X_stack[t].T, dnet_f)
            dWhf += np.matmul(self.Ht_stack[t-1].T, dnet_f)
            dBf += dnet_f

            #输入门
            dWxi += np.matmul(self.X_stack[t].T, dnet_i)
            dWhi += np.matmul(self.Ht_stack[t-1].T, dnet_i)
            dBi += dnet_i

            #Ht-1和Ct-1
            dH_1 = np.matmul(dnet_cct, self.Whc) + np.matmul(dnet_i, self.Whi) + np.matmul(dnet_f, self.Whf) + np.matmul(dnet_o, self.Who)
            dnet_ct_1 = dnet_ct * self.ft_stack[t]

        '''''
        #梯度裁剪(这里的限制范围需要自己根据需求调整，否则梯度太大会很难很难训练，loss会降不下去的)
        for dparam in [dWxh, dWhh, dBh, dWxz, dWhz, dBz, dWxr, dWhr, dBr]: 
            np.clip(dparam, -0.5, 0.5, out=dparam)#限制在[-0.5,0.5]之间
            #print(dparam)
        '''''

        #候选记忆细胞
        self.Wxc += -lr * dWxc
        self.Whc += -lr * dWhc
        self.B_c += -lr * dBc
        #输出门
        self.Wxo += -lr * dWxo
        self.Who += -lr * dWho
        self.B_o += -lr * dBo
        #遗忘门
        self.Wxf += -lr * dWxf
        self.Whf += -lr * dWhf
        self.B_f += -lr * dBf
        #输入门
        self.Wxi += -lr * dWxi
        self.Whi += -lr * dWhi
        self.B_i += -lr * dBi

        return self.loss

    def pre(self,input_onehot,h_prev,c_prev,next_len,vocab): #input_onehot为输入的一个词的onehot编码，next_len为需要生成的单词长度，vocab是"索引-词"的词典
        xs, hs, cs = {}, {}, {} #字典形式存储
        hs[-1] = np.copy(h_prev) #隐藏状态赋予
        cs[-1] = np.copy(c_prev)
        xs[0] = input_onehot
        pre_vocab = []
        for t in range(next_len):
            #输入门
            net_i = np.matmul(xs[t], self.Wxi) + np.matmul(hs[t-1], self.Whi) + self.B_i
            it = sigmoid(net_i)
            #遗忘门
            net_f = np.matmul(xs[t], self.Wxf) + np.matmul(hs[t-1], self.Whf) + self.B_f
            ft = sigmoid(net_f)
            #输出门
            net_o = np.matmul(xs[t], self.Wxo) + np.matmul(hs[t-1], self.Who) + self.B_o
            ot = sigmoid(net_o)
            #候选记忆细胞
            net_cc = np.matmul(xs[t], self.Wxc) + np.matmul(hs[t-1], self.Whc) + self.B_c
            cct = tanh(net_cc)
            #记忆细胞
            Ct = ft*cs[t-1]+it*cct
            cs[t] = Ct
            #隐藏状态
            Ht = ot*tanh(Ct)
            hs[t] = Ht
            #输出
            Ot = np.matmul(Ht, self.W_hd) + self.B_d
            Yt = np.exp(Ot) / np.sum(np.exp(Ot)) #softmax
            pre_vocab.append(vocab[np.argmax(Yt)])

            xs[t+1] = np.zeros((1, self.input_size)) # init
            xs[t+1][0,np.argmax(Yt)] = 1
        return pre_vocab
