"""
Created for Graduation Dissertation
This is a test which shows the W and b without the accuracy
Author:沈栩烽
"""

import tensorflow as tf
import numpy as np
import pandas as pd

tf.compat.v1.disable_eager_execution()

#数据预处理
f = open('49_EXL.csv', encoding='utf-8')
c1 = f.readlines()
head = c1[4:6]
h1 = head[0].split(',')
h2 = head[1].split(',')
for i in range(len(h1)):
    if h1[i] == '':
        h1[i] = h2[i]
header = h1

data_set = pd.read_csv('49.csv', header=None, names=[i for i in header])

#训练集和测试集的划分
train_set = pd.read_csv('49.csv', nrows = 800, header=None, names=[i for i in header])
test_set = data_set.drop(i for i in range(800))

maxA = np.max(train_set['ASH55-92'])
minA = np.min(train_set['ASH55-92'])
maxP = np.max(train_set['PMV'])
minP = np.min(train_set['PMV'])
#神经网络
x_data = np.array([i for i in train_set['ASH55-92']])
y_data = np.array([j for j in train_set['PMV']])


x_data_temp = []
y_data_temp = []
for i in range(800):
    x_data_temp.append(x_data[i]/(maxA - minA))

for j in range(800):
    y_data_temp.append(y_data[j]/(maxP - minP))

x_te = np.array([i for i in test_set['ASH55-92']])
y_te = np.array([j for j in test_set['PMV']])
#
x_data_new = np.array([i for i in x_data_temp])
y_data_new = np.array([j for j in y_data_temp])

W = tf.Variable(0.)
b = tf.Variable(0.)

y = W*x_data_new + b
loss = tf.reduce_mean(tf.square(y_data_new - y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.2)

train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([W, b]))


"""
指标 CLO
"""