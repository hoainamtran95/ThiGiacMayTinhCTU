import numpy as np
from scipy import signal, ndimage


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

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
    # kernel = np.flip(kernel, 1)
    margin = Wk // 2
    kernel = np.flip(kernel, 1)
    for x in range(margin, image.shape[0] - margin):
        for y in range(margin, image.shape[1] - margin):
            for i in range(-margin, margin + 1):
                for j in range(-margin, margin + 1):
                    out[x, y] = out[x, y] + image[x + i, y + j] * kernel[i + margin][j + margin]

    ### END YOUR CODE

    return out


# kernel = np.array([[2,4,2]]).T
# print('mat na :\n',kernel)
# A=np.matrix([[1,2,3],[3,2,1],[2,2,2],[1,1,1]])
# print("ma tran :\n",A)
# print("sau khi chap :\n",conv_nested(A,kernel))
# kernel = np.array(
# [
#     [1,0,1],
#     [0,0,0],
#     [1,0,1],
# ])
# test_img = np.full((9, 9),2)
# print(conv_nested(test_img, kernel))

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    ### YOUR CODE HERE
    out[pad_height:-pad_height, pad_width:-pad_width] = image[:, :]
    ### END YOUR CODE
    return out


# test=np.array([[1]])
# zero_pad(test,1,2)

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
    m = Hk // 2
    kernel_temp = np.copy(kernel)
    kernel_temp = np.flip(kernel_temp, 1)
    img=np.copy(image)
    for x in range(1, Hi - 1):
        for y in range(1, Wi - 1):
            out[x][y] = (kernel_temp * img[x - m:(x + m + 1), y - m:(y + m + 1)]).sum()

    ### END YOUR CODE

    return out
#
#
# kernel = np.array(
#     [
#         [1, 4, 6, 4, 1],
#         [4, 16, 24, 16, 4],
#         [6, 24, 36, 24, 6],
#         [4, 16, 24, 16, 4],
#         [1, 4, 6, 4, 1]
#     ])
# kernel2= np.array(
#         [
#             [-1, -1, -1],
#             [-1, 8, -1],
#             [-1, -1, -1]
#         ])
# A = np.array(
#     [
#         [1, 4, 6, 4, 1, 4, 16, 24, 16, 4],
#         [4, 16, 24, 16, 4, 4, 16, 24, 16, 4],
#         [6, 24, 36, 24, 6, 4, 16, 24, 16, 4],
#         [4, 16, 24, 16, 4, 4, 16, 24, 16, 4],
#         [1, 4, 6, 4, 1, 4, 16, 24, 16, 4],
#         [4, 16, 24, 16, 4, 4, 16, 24, 16, 4],
#         [6, 24, 36, 24, 6, 4, 16, 24, 16, 4],
#         [4, 16, 24, 16, 4, 4, 16, 24, 16, 4]
#     ])
# print(conv_fast(A, kernel2))
#

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
    out = signal.convolve2d(image, kernel_temp, mode='valid')
    ### END YOUR CODE
    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    # kernel = np.array(
    #     [
    #         [-1, -1, -1],
    #         [-1, 8, -1],
    #         [-1, -1, -1]
    #     ])
    kernel = np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
    f = conv_fast(f, kernel)
    g = conv_fast(g, kernel)
    temp = signal.correlate2d(f, g, 'same')
    out = np.copy(temp)

    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    mean = g.sum() / (g.shape[0] * g.shape[1])
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            g[i, j] = 0 if g[i, j] == mean else g[i, j]

    out = cross_correlation(f, g)
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
    k1 = np.ones((3, 3))
    k1[1, 1] = 2
    k2 = np.full((3, 3), -1 / 9)
    print(k1)
    print(k2)
    self1 = conv_fast(f, k1)
    self2 = conv_fast(self1, k2)
    out = self2

    ### END YOUR CODE
    return out
