import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io
from scipy import signal, ndimage
import math


# u = np.array([[-0.1329, 0.9581, -0.2537]
#                  , [-0.9822, -0.1617, -0.0959]
#                  , [-0.1329, 0.2364, 0.9625]])
# s = np.array([
#     [0.6420, 0, 0]
#     , [0, 0.0000, 0]
#     , [0, 0, 0.0000]])
# v = np.array([[-0.1329, -0.6945, -0.7071]
#              , [-0.9822, 0.1880, 0.0000]
#              , [-0.1329, -0.6945, 0.7071]])

#
# gaussian = np.array(
# [
#     [1,4,6,4,1],
#     [4,16,24,16,4],
#     [6,24,36,24,6],
#     [4,16,24,16,4],
#     [1,4,6,4,1]
# ])
# k1 = None  # shape (5, 1)
# k2 = None  # shape (1, 5)
#
# u,s,v=np.linalg.svd(gaussian)
#
# sqrt_value = math.sqrt(s[0])
# k1 = u[:, 0] * sqrt_value
# k2 = v[:, 0] * sqrt_value
#
# k1 = np.array([k1]).T
# k2 = np.array([k2])
#
#
# print(k1)
# print(k2)
# print(k1*k2)
def display(img):
    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)

    ### END YOUR CODE

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel_temp = np.copy(kernel)
    out = signal.convolve2d(image, kernel_temp, mode='same')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel_temp = np.copy(kernel)
    kernel_temp = np.flip(kernel_temp, 1)
    for x in range(1, Hi - 1):
        for y in range(1, Wi - 1):
            out[x][y] = (kernel_temp * image[x - 1:x + 2, y - 1:y + 2]).sum()

    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    out = None
    ### YOUR CODE HERE
    # k1=np.ones((3,3))
    # k1[1,1]=2
    # k2=np.full((3,3),-1/9)
    # print(k1)
    # print(k2)
    # self1=conv_fast(f,k1)
    # self2=conv_fast(self1,k2)
    # out=self2

    im = ndimage.gaussian_filter(f, 8)

    sx = ndimage.sobel(f, axis=0, mode='constant')
    sy = ndimage.sobel(f, axis=1, mode='constant')
    sob = np.hypot(sx, sy)

    ### END YOUR CODE
    out = sob
    return out


shelve = io.imread("C:\\Users\\quocb14005xx\\Documents\\Python Scripts\\thigiacmaytinh_trenlop\\hw1\\shelf_dark.jpg",
                   as_grey=True)
item = io.imread("C:\\Users\\quocb14005xx\\Documents\\Python Scripts\\thigiacmaytinh_trenlop\\hw1\\template.jpg",
                 as_grey=True)

kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
shelve=conv_faster(shelve,kernel)
item=conv_faster(item,kernel)
a=signal.correlate2d(shelve,item,'same')
display(a)
