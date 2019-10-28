import time
import torch
import imutil
from torchvision import datasets
from sklearn.decomposition import NMF


# Try playing around with different numbers of components
n_components = 16

# eg. digit=7 will factorize only sevens
digit = None


def get_mnist(digit=None):
    data = datasets.mnist.MNIST('./data', download=True)
    if digit:
        mnist = data.train_data[data.train_labels == digit]
    else:
        mnist = data.train_data
    mnist = mnist.type(torch.FloatTensor)
    return mnist


def get_filename():
    return 'output-{}.jpg'.format(int(time.time()*1000))


def factorize(data, width=28):
    nmf = NMF(n_components=n_components, init='random', random_state=0)
    W = nmf.fit_transform(data.reshape(-1, width*width))
    H = nmf.components_
    return H.reshape(-1, width, width)


def main():
    print('Starting NMF demo')

    print('Downloading MNIST data...')
    mnist = get_mnist(digit=None)
    filename = get_filename()
    imutil.show(mnist.mean(0), filename=filename, resize_to=(256,256), caption="Average MNIST digit")
    print('Output average MNIST digit to {}'.format(filename))

    print('Computing factorization, please wait...')
    start_time = time.time()
    clusters = factorize(mnist)
    print('Completed factorization with {} components in {:.3f}s'.format(n_components, time.time() - start_time))

    filename = get_filename()
    caption = "Summarizing {} samples in {} clusters".format(len(mnist), n_components)
    imutil.show(clusters, filename=filename, resize_to=(512,512), caption=caption)
    print('Wrote output file to {}'.format(filename))


if __name__ == '__main__':
    main()
