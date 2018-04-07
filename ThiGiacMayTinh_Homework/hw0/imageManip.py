import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

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

def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = image
    for x in range(out.shape[0]):
        for y in range(out.shape[1]):
            temp = out[x][y] ** 2
            out[x][y] = 0.5 * temp
    ### END YOUR CODE

    return out


# a=load("image1.jpg")
# b=change_value(a)
# cv2.imshow("b",b)
# cv2.waitKey(0)

def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    out = image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            out[x][y] = 0.299 * out[x][y][0] + 0.587 * out[x][y][1] + 0.114 * out[x][y][2]
    ### END YOUR CODE

    return out


#
# a=load("image1.jpg")
# b=convert_to_grey_scale(a)
# cv2.imshow("b",b)
# cv2.waitKey(0)
def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    ### YOUR CODE HERE
    out = np.copy(image)
    if channel in str('R'):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out[i, j][0] = 0
    if channel in str('G'):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out[i, j][1] = 0
    if channel in str('B'):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out[i, j][2] = 0
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = np.copy(lab)

    ### YOUR CODE HERE
    if channel == str('L'):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out[i, j][0] = 0
    if channel == str('A'):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out[i, j][1] = 0
    if channel == str('B'):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                out[i, j][2] = 0
    ### END YOUR CODE
    return out


def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = np.copy(hsv)

    ### YOUR CODE HERE
    # if channel== 'H':
    #     out[0:out.shape[0],0:out.shape[1]][0] = 0
    # if channel== 'S':
    #     out[0:out.shape[0],0:out.shape[1]][1] = 0
    # if channel== 'V':
    #     out[0:out.shape[0],0:out.shape[1]][2] = 0
    if channel == str('H'):
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                out[x][y][0] = 0
    if channel == str('S'):
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                out[x][y][1] = 0
    if channel == str('V'):
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                out[x][y][2] = 0
                ### END YOUR CODE

    return out



def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = image1
    ### YOUR CODE HERE
    left_index=int(image1.shape[1] / 2)
    left_img=image1[:,0:left_index]

    right_index=int(image2.shape[1] / 2)
    right_img=image2[:,right_index:image2.shape[1]]


    left_img=rgb_decomposition(left_img,channel1)
    right_img=rgb_decomposition(right_img,channel2)


    np.resize(out,(image1.shape[0],left_img.shape[1]+right_img.shape[1]))

    out[:,0:(image1.shape[1]//2)]=left_img
    out[:,(image2.shape[1]//2):image2.shape[1]]=right_img


    ### END YOUR CODE

    return out


# mix_images(load("image1.jpg"),load("image2.jpg"),'B','R')