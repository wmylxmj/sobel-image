# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:15:34 2018

@author: wmy
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import PIL
from math import sqrt, e, exp

myimg = mpimg.imread('timg_16.jpg')
plt.imshow(myimg)
plt.axis('on')
plt.show()
print(myimg.shape)

full_shape = []
full_shape.append(1)
full_shape.append(myimg.shape[0])
full_shape.append(myimg.shape[1])
full_shape.append(myimg.shape[2])
print(full_shape)
full = np.reshape(myimg, full_shape)

inputfull = tf.Variable(tf.constant(1.0, shape = full_shape))


sobel = [[-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0],
         [-2.0, -2.0, -2.0],
         [0.0, 0.0, 0.0],
         [2.0, 2.0, 2.0],
         [-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0]]

test = [[-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0],
         [-sqrt(2.0), -sqrt(2.0), -sqrt(2.0)],
         [0.0, 0.0, 0.0],
         [sqrt(2.0), sqrt(2.0), sqrt(2.0)],
         [-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0]]

test2 = [[-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0],
         [-sqrt(e), -sqrt(e), -sqrt(e)],
         [0.0, 0.0, 0.0],
         [sqrt(e), sqrt(e), sqrt(e)],
         [-1.0, -1.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0]]

test3 = [[1.0, 0.0, -1.0],
         [2.0, 0.0, -2.0],
         [1.0, 0.0, -1.0],
         [1.0, 0.0, -1.0],
         [2.0, 0.0, -2.0],
         [1.0, 0.0, -1.0],
         [1.0, 0.0, -1.0],
         [2.0, 0.0, -2.0],
         [1.0, 0.0, -1.0]]

test4 = [[1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [-1.0, -1.0, -1.0],
         [-2.0, -2.0, -2.0],
         [-1.0, -1.0, -1.0]]

test5 = [[-1.0, -1.0, -1.0],
         [0.0, -2.0, 0.0],
         [1.0, -1.0, 1.0],
         [-2.0, 0.0, -2.0],
         [0.0, 0.0, 0.0],
         [2.0, 0.0, 2.0],
         [-1.0, 1.0, -1.0],
         [0.0, 2.0, 0.0],
         [1.0, 1.0, 1.0]]

test6 = [[1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0]]

test7 = [[2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [2.0, 2.0, 2.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0]]

test8 = [[-1.0, -1.0, -1.0],
         [0.1, -2.0, 0.1],
         [1.0, -1.0, 1.0],
         [-2.0, 0.1, -2.0],
         [0.1, 0.1, 0.1],
         [2.0, 0.1, 2.0],
         [-1.0, 1.0, -1.0],
         [0.1, 2.0, 0.1],
         [1.0, 1.0, 1.0]]

test9 = [[-1.0, -1.0, -1.0],
         [0.1, -2.0, -2.0],
         [1.0, -1.0, -1.0],
         [-2.0, 0.1, 0.1],
         [0.1, 0.1, 0.1],
         [2.0, 0.1, 0.1],
         [-1.0, 1.0, 1.0],
         [0.1, 2.0, 2.0],
         [1.0, 1.0, 1.0]]

test10 =[[1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0],
         [-1.0, -1.0, -1.0],
         [1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0],
         [-1.0, -1.0, -1.0],
         [-1.0, -1.0, -1.0],
         [-1.0, -1.0, -1.0],
         [1.0, 1.0, 1.0]]

test11 =[[1.0, 1.0, 1.0],
         [-1.0, -1.0, -1.0],
         [-1.0, -1.0, -1.0],
         [1.0, 1.0, 1.0],
         [0.1, 0.1, 0.1],
         [-1.0, -1.0, -1.0],
         [-1.0, -1.0, -1.0],
         [1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]]

my_filter = tf.Variable(tf.constant(test11, shape = [3, 3, 3, 1]))
print(my_filter)
op = tf.nn.conv2d(inputfull, my_filter, strides=[1, 1, 1, 1], padding='SAME')

o = tf.cast(((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255, tf.uint8)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    t, f = sess.run([o, my_filter], feed_dict={inputfull : full})

    t = np.reshape(t, [myimg.shape[0], myimg.shape[1]])

    plt.imshow(t, cmap='Greys_r')
    plt.axis('on')
    plt.show()
    
    imagevector = np.array(t)
    height,width = imagevector.shape
    for h in range(height):
        for w in range(width):
            if imagevector[h,w]<=128:
                imagevector[h,w]=0
            else:
                imagevector[h,w]=1
                pass
            pass
        pass
    plt.imshow(imagevector,cmap='gray')
    plt.axis('on')
    plt.show()


