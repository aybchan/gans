import tensorflow as tf
import numpy as np
import utils
from ops import *

import time
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

def discriminator(x, reuse=False, name='d'):
    dim = 64

    with tf.variable_scope(name, reuse=reuse):
        h0 = leaky_relu(conv2d(x, 128, name='d_h0'))
        h1 = leaky_relu(batch_norm(conv2d(h0, 256, name='d_h1'),'d_bn1'))
        h2 =     dense(tf.layers.flatten(h1),1,name='d_d2')
        return tf.nn.sigmoid(h2),h2

def generator(z,batch_size,name='g',reuse=False):

    with tf.variable_scope(name,reuse=reuse):
        h0 = dense(z, 256*7*7, name='g_1')
        h0 = tf.reshape(h0, [batch_size, 7,7, 256])
        h0 = leaky_relu(batch_norm(h0, name='g_bn1'))

        h1 = dconv2d(h0, 128, batch_size, kernel=(5, 5), strides=(2, 2), name='g_2')
        h1 = leaky_relu(batch_norm(h1, name='g_bn2'))

        h2 = dconv2d(h1, 1, batch_size, kernel=(5, 5), strides=(2, 2), name='g_3')
        h2 = leaky_relu(batch_norm(h2, name='g_bn3'))
        return tf.nn.tanh(h2)

# Model train settings
load = False        # Change to True to load a saved model
save_dir = 'saved_model'  # Specify direction to save/load model from
epochs = 100
batch_size = 64

tf.reset_default_graph()

x = tf.placeholder(tf.float32,shape=[None,28,28,1])
z = tf.placeholder(tf.float32,shape=[None,100])

g = generator(z,batch_size)

d_loss_real,d_logit_real = discriminator(x)
d_loss_fake,d_logit_fake = discriminator(g,True)

d_params = [v for v in tf.trainable_variables() if v.name.startswith('d/')]
g_params = [v for v in tf.trainable_variables() if v.name.startswith('g/')]

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_logit_real,labels=tf.ones_like(d_logit_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_logit_fake,labels=tf.zeros_like(d_logit_fake)))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_logit_fake,labels=tf.ones_like(d_logit_fake)))
d_loss = d_loss_real + d_loss_fake

g_train = tf.train.AdamOptimizer(2e-4).minimize(g_loss,var_list=g_params)
d_train = tf.train.AdamOptimizer(2e-4).minimize(d_loss,var_list=d_params)

np.random.seed(2018)
code = np.random.normal(size=[batch_size,100])

with tf.Session() as sess:
    d_losses, g_losses = [], []
    steps  = epochs * mnist.train.num_examples
    saver = tf.train.Saver()

    if load:
        saver.restore(sess,save_dir + '/model.ckpt')
    else:
        init = tf.global_variables_initializer()
        init.run()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    print('Epoch\tDisc. loss\tGen. loss\tTime')
    for epoch in range(epochs):
        start = time.time()

        for batch in range(mnist.train.num_examples//batch_size):
            imgs = (mnist.train.next_batch(batch_size)[0]
						.reshape([-1,28,28,1]))
            sess.run(g_train,feed_dict={z: np.random.normal(
                                                    size=[batch_size,100])})
            sess.run(d_train,feed_dict={z: np.random.normal(
                                                    size=[batch_size,100]),
                                        x: imgs})
            if batch % 20 == 0:
                losses   = sess.run([d_loss,g_loss],
			feed_dict={z: np.random.normal(size=[batch_size,100]),
				   x: imgs})
                d_losses.append(losses[0])
                g_losses.append(losses[1])

        end = time.time()
        print('{0}\t{1:.4f}\t\t{2:.4f}\t\t{3:.1f}s\t'
                    .format(epoch,losses[0],losses[1],end-start)
                    + time.strftime("%X"))

        saver.save(sess,save_dir + '/model.ckpt')

        images = sess.run(g,feed_dict={z: code})
        step   = (epoch + 1) * mnist.train.num_examples
        utils.plot('outputs/',epoch,images,step,steps,[g_losses,d_losses])
