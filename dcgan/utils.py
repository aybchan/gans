import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tick
plt.style.use('fivethirtyeight')
import os

def plot(output_dir,epoch,images,step,steps,loss):
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
    def k(x,pos):
      x /= 1000.0
      return '%.1f%s' % (x, 'K')


    xs = np.linspace(0,step,len(loss[0]))
    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Epoch %d' % (epoch) , fontsize=20,x=0.55)

    gs1 = gridspec.GridSpec(8,8)
    images = images.reshape([64,28,28])
    for i,subplot in enumerate(gs1):
        ax = fig.add_subplot(subplot)
        ax.imshow(images[i],cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_axis_off()
    gs1.tight_layout(fig, rect=[0, 0, 0.5,1])
    gs1.update(wspace=0.0, hspace=0.0)

    gs2 = gridspec.GridSpec(2,1)

    c = ['#008FD5','#FF2700']
    title = ['Generator loss','Discriminator loss']

    for p in range(2):
        ax = fig.add_subplot(gs2[p])
        ax.plot(xs,loss[p], linewidth=1.5,alpha=0.3,c=c[p])
        ax.plot(xs,savitzky_golay(loss[p],61,5),c=c[p])
        ax.set_title(title[p],fontsize=12)
        ax.set_xlabel('Step',fontsize=10)
        ax.set_ylabel('Loss',fontsize=10)
        ax.set_xlim([0,steps])
        ax.xaxis.set_major_formatter(tick.FuncFormatter(k))

    gs2.tight_layout(fig, rect=[0.5, 0, 1, 1])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = output_dir + str(epoch).zfill(3)+ '.png'
    plt.savefig(file_name)

