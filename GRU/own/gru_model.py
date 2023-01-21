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

class GRU(object):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        #重置门
        self.Wxr = np.random.randn(input_size, hidden_size)
        self.Whr = np.random.randn(hidden_size, hidden_size)
        self.B_r  = np.zeros((1, hidden_size))
        #更新门
        self.Wxz = np.random.randn(input_size, hidden_size)
        self.Whz = np.random.randn(hidden_size, hidden_size)
        self.B_z = np.zeros((1, hidden_size))
        #候选隐藏状态
        self.Wxh = np.random.randn(input_size, hidden_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.B_h = np.zeros((1, hidden_size))
        #输出
        self.W_o = np.random.randn(hidden_size, input_size)
        self.B_o = np.zeros((1, input_size))

    def forward(self,X,Ht_1): #前向传播
        #存储
        self.rt_stack = {} #重置门存储
        self.zt_stack = {} #更新门存储
        self.hht_stack = {} #候选隐藏状态存储
        self.X_stack = {} #X存储
        self.Ht_stack = {} #隐藏状态存储
        self.Y_stack = {} #输出存储

        self.Ht_stack[-1] = Ht_1
        self.T = X.shape[0]

        for t in range(self.T):
            self.X_stack[t] = X[t].reshape(-1,1).T
            #重置门
            net_r = np.matmul(self.X_stack[t], self.Wxr) + np.matmul(self.Ht_stack[t-1], self.Whr) + self.B_r
            rt = sigmoid(net_r)
            self.rt_stack[t] = rt
            #更新门
            net_z = np.matmul(self.X_stack[t], self.Wxz) + np.matmul(self.Ht_stack[t-1], self.Whz) + self.B_z
            zt = sigmoid(net_z)
            self.zt_stack[t] = zt
            #候选隐藏状态
            net_hh = np.matmul(self.X_stack[t], self.Wxh) + np.matmul(rt*self.Ht_stack[t-1], self.Whh) + self.B_h
            hht = tanh(net_hh)
            self.hht_stack[t] = hht
            #隐藏状态
            Ht = zt*self.Ht_stack[t-1] + (1-zt)*hht
            self.Ht_stack[t] = Ht
            #输出
            Ot = np.matmul(Ht, self.W_o) + self.B_o
            Yt = np.exp(Ot) / np.sum(np.exp(Ot)) #softmax
            self.Y_stack[t] = Yt

    def backward(self,target,lr):
        #初始化
        dW_o, dB_o, dH, dH_1 = np.zeros_like(self.W_o), np.zeros_like(self.B_o), np.zeros_like(self.Ht_stack[-1]), np.zeros_like(self.Ht_stack[-1])

        dWxh, dWhh, dBh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.B_h)
        dWxr, dWhr, dBr = np.zeros_like(self.Wxr), np.zeros_like(self.Whr), np.zeros_like(self.B_r)
        dWxz, dWhz, dBz = np.zeros_like(self.Wxz), np.zeros_like(self.Whz), np.zeros_like(self.B_z)

        self.loss = 0

        for t in reversed(range(self.T)): #反过来开始，因为像隐藏状态求偏导那样，越往前面分支越多       
            dY = self.Y_stack[t] - target[t].reshape(-1,1).T
            self.loss += -np.sum(np.log(self.Y_stack[t]) * target[t].reshape(-1,1).T)
            #对输出的参数
            dW_o += np.matmul(self.Ht_stack[t].T,dY)
            dB_o += dY

            dH = np.matmul(dY, self.W_o.T) + dH_1 #dH更新

            #对有关更新门，重置门，候选隐藏状态中参数的求导的共同点
            dnet_hht = dH * (1-self.zt_stack[t]) * (1-self.hht_stack[t] * self.hht_stack[t]) #候选隐藏状态
            dnet_Z = dH * (self.Ht_stack[t-1] - self.hht_stack[t]) * self.zt_stack[t] *(1 - self.zt_stack[t]) #更新门
            dnet_R = np.matmul(dnet_hht, self.Whh) * self.Ht_stack[t-1] * self.rt_stack[t] *(1 - self.rt_stack[t]) #重置门

            #候选隐藏状态中参数
            dWxh += np.matmul(self.X_stack[t].T, dnet_hht)
            dWhh += np.matmul((self.rt_stack[t]*self.Ht_stack[t-1]).T, dnet_hht)
            dBh += dnet_hht

            #更新门
            dWxz += np.matmul(self.X_stack[t].T, dnet_Z)
            dWhz += np.matmul(self.Ht_stack[t-1].T, dnet_Z)
            dBz += dnet_Z

            #重置门
            dWxr += np.matmul(self.X_stack[t].T, dnet_R)
            dWhr += np.matmul(self.Ht_stack[t-1].T, dnet_R)
            dBr += dnet_R

            #Ht-1
            dH_1 = dH * self.zt_stack[t] + np.matmul(dnet_hht, self.Whh) * self.rt_stack[t] + np.matmul(dnet_R, self.Whr) + np.matmul(dnet_Z, self.Whz)

        '''''
        #梯度裁剪(这里的限制范围需要自己根据需求调整，否则梯度太大会很难很难训练，loss会降不下去的)
        for dparam in [dWxh, dWhh, dBh, dWxz, dWhz, dBz, dWxr, dWhr, dBr]: 
            np.clip(dparam, -0.5, 0.5, out=dparam)#限制在[-0.5,0.5]之间
            #print(dparam)
        '''''

        #候选隐藏状态
        self.Wxh += -lr * dWxh
        self.Whh += -lr * dWhh
        self.B_h += -lr * dBh
        #更新门
        self.Wxz += -lr * dWxz
        self.Whz += -lr * dWhz
        self.B_z += -lr * dBz
        #重置门
        self.Wxr += -lr * dWxr
        self.Whr += -lr * dWhr
        self.B_r += -lr * dBr

        return self.loss

    def pre(self,input_onehot,h_prev,next_len,vocab): #input_onehot为输入的一个词的onehot编码，next_len为需要生成的单词长度，vocab是"索引-词"的词典
        xs, hs = {}, {} #字典形式存储
        hs[-1] = np.copy(h_prev) #隐藏变量赋予
        xs[0] = input_onehot
        pre_vocab = []
        for t in range(next_len):
            #重置门
            net_r = np.matmul(xs[t], self.Wxr) + np.matmul(hs[t-1], self.Whr) + self.B_r
            rt = sigmoid(net_r)
            #更新门
            net_z = np.matmul(xs[t], self.Wxz) + np.matmul(hs[t-1], self.Whz) + self.B_z
            zt = sigmoid(net_z)
            #候选隐藏状态
            net_hh = np.matmul(xs[t], self.Wxh) + np.matmul(rt*hs[t-1], self.Whh) + self.B_h
            hht = tanh(net_hh)
            #隐藏状态
            hs[t] = zt*hs[t-1] + (1-zt)*hht
            #输出
            Ot = np.matmul(hs[t], self.W_o) + self.B_o
            Yt = np.exp(Ot) / np.sum(np.exp(Ot)) #softmax
            pre_vocab.append(vocab[np.argmax(Yt)])

            xs[t+1] = np.zeros((1, self.input_size)) # init
            xs[t+1][0,np.argmax(Yt)] = 1
        return pre_vocab
