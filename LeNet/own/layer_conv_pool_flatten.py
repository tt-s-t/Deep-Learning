import numpy as np

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding=0, stride=1):
        # 卷积层各参数
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.init_param()

    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        # weight: [cin,k,k,cout]
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):  # 前向传播的计算
        self.input = input # [N, C, H, W]
        # 边界扩充
        height = self.input.shape[2] + self.padding * 2#输入的最终高
        width = self.input.shape[3] + self.padding * 2#输入的最终宽
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])#初始化，定好input的最终的shape
        self.input_pad[:,:,self.padding:self.padding+self.input.shape[2],self.padding:self.padding+self.input.shape[3]] = self.input#填充为0，其余不变
        #输出的特征图的高宽（Nout=(Nin+2p-k)/s+1）
        height_out = (height-self.kernel_size)//self.stride+1
        width_out = (width-self.kernel_size)//self.stride+1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):#先左右后上下
                    for idxw in range(width_out):
                        # 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * self.input_pad[idxn, :, hs:hs+self.kernel_size, ws:ws+self.kernel_size]) + self.bias[idxc]
        return self.output

    def backward(self,top_diff,lr): # 反向传播的计算
        #初始化卷积核，偏置以及累计到本层的梯度
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)

        #计算dW，db，dX。top_diff————（在前向传播中相对该层是输出的梯度）传递下来
        for idxn in range(top_diff.shape[0]):#N
            for idxc in range(top_diff.shape[1]):#C
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        # 对W
                        self.d_weight[:, :, :, idxc] += np.dot(top_diff[idxn,idxc,idxh,idxw],self.input_pad[idxn,:,hs:hs+self.kernel_size, ws:ws+self.kernel_size])

                        # 对b
                        self.d_bias[idxc] += top_diff[idxn,idxc,idxh,idxw]

                        #对X
                        bottom_diff[idxn, :, hs:hs+self.kernel_size, ws:ws+self.kernel_size] += top_diff[idxn,idxc,idxh,idxw] * self.weight[:,:,:,idxc]
        #减去padding的部分
        bottom_diff = bottom_diff[:,:,self.padding:bottom_diff.shape[2]-self.padding,self.padding:bottom_diff.shape[3]-self.padding]

        self.update_param(lr)#更新参数W和b
        return bottom_diff

    def update_param(self, lr): #更新参数W和b
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input): # 前向传播的计算
        self.input = input # [N, C, H, W]
        height_out = (self.input.shape[2]-self.kernel_size)//self.stride+1
        width_out = (self.input.shape[3]-self.kernel_size)//self.stride+1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
			            # 计算最大池化层的前向传播， 取池化窗口内的最大值
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc,hs:hs + self.kernel_size,ws:ws + self.kernel_size])
        return self.output
    
    def backward(self,top_diff): # 后向传播的计算
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]): #N
            for idxc in range(top_diff.shape[1]): #C
                for idxh in range(top_diff.shape[2]): #H
                    for idxw in range(top_diff.shape[3]): #W
                        max_index = np.unravel_index(
                            np.argmax(self.input[idxn, idxc,
                                      idxh * self.stride:idxh * self.stride + self.kernel_size,
                                      idxw * self.stride:idxw * self.stride + self.kernel_size])
                            , [self.kernel_size, self.kernel_size])#np.argmax()得到的最大值索引是将input展平后得到的索引，再利用np.unravel_index()得到在（k，k）中对应的最大值索引
                        
                        bottom_diff[idxn, idxc, idxh * self.stride + max_index[0], idxw * self.stride + max_index[1]] =top_diff[idxn, idxc, idxh, idxw]#只有对应选中的元素才有梯度传递，其他的被舍弃了
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):  # 扁平化层的初始化
        #这里的参数Input_shape和output_shape（通常是一个数）是针对一个样本而言的shape，即忽略N
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)#检验他们的大小是否一样大，才能进行对应的shape变化

    def forward(self, input): # 前向传播的计算
        assert list(input.shape[1:]) == list(self.input_shape)#进行验证输入是否与设定一致
        # 转换 input 维度顺序 (N,C,H,W)->(N,H,W,C)
        self.input = np.transpose(input,[0,2,3,1])
        self.output = self.input.reshape(self.input.shape[0],self.output_shape[0])#self.input.shape[0]指代N，self.output_shape[0]——H*W*C
        return self.output

    def backward(self,top_diff): # 后向传播的计算
        assert list(top_diff.shape[1:]) == list(self.output_shape)#进行验证上层梯度传递的shape是否与设定一致 
        #top_diff = np.transpose(top_diff, [0, 3, 1, 2])#调整顺序(N,H,W,C)-> (N,C,H,W)
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))#调整shape
        return bottom_diff


