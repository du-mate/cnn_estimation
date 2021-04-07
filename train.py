import pretty_errors
import os
from keras.layers import Input,Conv2D
from keras.models import Model
from sklearn.metrics import mean_squared_error
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


train_data=[]
label=[]
test_data=[]
test_label=[]

for file in os.listdir('dataset_2w/train_data'):    #训练数据
    train_data.append('dataset_2w/train_data'+'/'+file)

for file in os.listdir('dataset_2w/label'): #训练标签
    label.append('dataset_2w/label'+'/'+file)

for file in os.listdir('dataset_2w/test_data'): #测试数据
    test_data.append('dataset_2w/test_data'+'/'+file)

for file in os.listdir('dataset_2w/test_label'): #测试标签
    test_label.append('dataset_2w/test_label'+'/'+file)

train_data.sort(reverse=False)
label.sort(reverse=False)
test_data.sort(reverse=False)
test_label.sort(reverse=False)


train_data_array=np.random.rand(16000,49,49)
label_array=np.random.rand(16000,49,49)
test_data_array=np.random.rand(4000,49,49)
test_label_array=np.random.rand(4000,49,49)

for i in range(16000): #训练集
    f=h5py.File(train_data[i],'r')
    a=f['DS'][:]
    train_data_array[i]=np.array(a)

    f = h5py.File(label[i], 'r')
    a = f['DS'][:]
    label_array[i] = np.array(a)

for i in range(4000): #测试集
    f=h5py.File(test_data[i],'r')
    a=f['DS'][:]
    test_data_array[i]=np.array(a)

    f = h5py.File(test_label[i], 'r')
    a = f['DS'][:]
    test_label_array[i] = np.array(a)





input_img=Input(shape=(49,49,1))   #612导频*14符号

#编码
x=Conv2D(filters=64,kernel_size=(9,9),strides=(1, 1),padding='same',activation='relu')(input_img)
print(x.shape)  #(None,49,49,64)

x=Conv2D(filters=64,kernel_size=(5,5),strides=(1, 1),padding='same',activation='relu')(x)
print(x.shape)  #(None,49,49,64)

x=Conv2D(filters=64,kernel_size=(5,5),strides=(1, 1),padding='same',activation='relu')(x)
print(x.shape)

x=Conv2D(filters=32,kernel_size=(5,5),strides=(1, 1),padding='same',activation='relu')(x)
print(x.shape)

decode=Conv2D(filters=1,kernel_size=(5,5),strides=(1, 1),padding='same',activation='relu')(x)
print(decode.shape) #(None,49,49,1)

Model(inputs=input_img,outputs=decode)
autoencoder=Model(input_img,decode)
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

autoencoder.fit(train_data_array,label_array,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(test_data_array,test_label_array)
                )

#查看结果
predict_data=autoencoder.predict(test_data_array)   #shape=(20,49,49,1)



sum_neural_MSE=0
sum_interpolation_MSE=0
predict_data=predict_data.reshape(4000,49,49)

for i in range(10):
    sum_neural_MSE+=mean_squared_error(test_label_array[i],predict_data[i])
    sum_interpolation_MSE+=mean_squared_error(test_label_array[i],test_data_array[i])

print(sum_neural_MSE/10)
print(sum_interpolation_MSE/10)



n=10

for i in range(n):
    ax=plt.subplot(3,n,i+1) #插值
    plt.imshow(test_data_array[i])  #数据化为图片格式
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax=plt.subplot(3,n,i+1+n)   #真实
    plt.imshow(test_label_array[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i+1+2*n)   #神经网络
    plt.imshow(predict_data[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()























