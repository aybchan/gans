import tensorflow as tf
import numpy as np

import datetime
import time
import os

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
plt.style.use('fivethirtyeight')

def plot(output_dir,epoch,images,step,steps,g_losses,d_losses):
    # line smoothing for plotting loss
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        import numpy as np
        from math import factorial

        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        b = np.mat([[k**i for i in order_range] for k
                                        in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')
    xx = np.linspace(0,step,len(g_losses))
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Epoch %d' % (epoch) , fontsize=20,x=0.53)

    gs1 = gridspec.GridSpec(8,8)
    images = images.reshape([64,28,28])
    for i,subplot in enumerate(gs1):
        ax = fig.add_subplot(subplot)
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_axis_off()
    gs1.tight_layout(fig, rect=[0, 0, 0.5,1])
    gs1.update(wspace=0.0, hspace=0.0)

    gs2 = gridspec.GridSpec(2,1)
    ax1 = fig.add_subplot(gs2[0])
    ax1.plot(xx,g_losses,linewidth=1.5,alpha=0.3,c='#008FD5')
    ax1.plot(xx,savitzky_golay(g_losses,61,5),c='#008FD5')
    ax1.set_title('Generator loss',fontsize=12)
    ax1.set_xlabel('Step',fontsize=10)
    ax1.set_ylabel('Loss',fontsize=10)
    ax1.set_xlim([0,steps])

    ax2 = fig.add_subplot(gs2[1])
    ax2.plot(xx,d_losses,linewidth=1.5,alpha=0.3,c='#FF2700')
    ax2.plot(xx,savitzky_golay(d_losses,61,5),c='#FF2700')
    ax2.set_title('Discriminator loss',fontsize=12)
    ax2.set_xlabel('Step',fontsize=10)
    ax2.set_ylabel('Loss',fontsize=10)
    ax2.set_xlim([0,steps])

    gs2.tight_layout(fig, rect=[0.5, 0, 1, 1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = output_dir + str(epoch).zfill(3)+ '.png'
    plt.savefig(file_name)

def discriminator(x,reuse=False,name='d'):
    with tf.variable_scope(name, reuse=reuse):
        h0 = tf.layers.dense(x,256,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        h0 = tf.nn.relu(h0)

        h1 = tf.layers.dense(h0,1,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        out = tf.nn.sigmoid(h1)
        return out,h1

def generator(z,reuse=False,name='g'):
  with tf.variable_scope(name, reuse=reuse):
    h0 = tf.layers.dense(z,256,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    h0 = tf.nn.dropout(tf.nn.relu(h0),keep_prob=0.5)

    h1 = tf.layers.dense(h0,784,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    out = tf.nn.sigmoid(h1)
    return out

tf.reset_default_graph()

x = tf.placeholder(tf.float32,shape=[None,784])
z = tf.placeholder(tf.float32,shape=[None,100])

g = generator(z)

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

# Model train settings
load = False        # Change to True to load a saved model
save_dir = 'saved_model'  # Specify direction to save/load model from
epochs = 100
batch_size = 64

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
            imgs = (mnist.train.next_batch(batch_size)[0])
            sess.run(g_train,feed_dict={z: np.random.normal(
                                                    size=[batch_size,100])})
            sess.run(d_train,feed_dict={z: np.random.normal(
                                                    size=[batch_size,100]),
                                        x: imgs})
            if batch % 20 == 0:
                r   = sess.run([d_loss,g_loss],feed_dict={z: code,x: imgs})
                d_losses.append(r[0])
                g_losses.append(r[1])
        saver.save(sess,save_dir + '/model.ckpt')
        end = time.time()
        losses = sess.run([d_loss,g_loss],feed_dict={z: code,x: imgs})
        print('{0}\t{1:.4f}\t\t{2:.4f}\t\t{3:.1f}s'
                            .format(epoch,losses[0],losses[1],end-start))

        images = sess.run(g,feed_dict={z: code})
        step   = (epoch + 1) * mnist.train.num_examples
        plot('outputs/',epoch,images,step,steps,g_losses,d_losses)
