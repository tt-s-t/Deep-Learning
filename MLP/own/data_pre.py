import os
import struct
import numpy as np

def load_mnist(file_dir, is_images):#is_images为True时加载image,False加载labels
    # Read binary data（二进制文件读取）
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header（根据要读的文件类别，进行区分——图像和lables）
    #首先读取数据信息
    if is_images:
        # Read images
        fmt_header = '>iiii'
        _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)#将缓冲区bin_data中的内容在按照指定的格式fmt='fmt_header'，从偏移量为offset=0的位置开始进行读取。返回的是一个对应的元组tuple
    else:
        # Read labels
        fmt_header = '>ii'
        _, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    #读取数据本体内容
    data_size = num_images * num_rows * num_cols#数据大小
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))#读取数据内容
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])#对于图像数据：(28,28)变784
    return mat_data
    
def load_data(data_path, label_path):
    images = load_mnist(data_path, True)
    labels = np.squeeze(load_mnist(label_path, False))
    return images,labels