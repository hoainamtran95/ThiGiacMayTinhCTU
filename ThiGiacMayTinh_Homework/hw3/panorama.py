import numpy as np
import scipy
from skimage import filters
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from utils import pad, unpad


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))
    k = int((size - 1) / 2)
    ### YOUR CODE HERE
    for i in range(
            size):  # do cong thuc toan hoc la i j ma index trong toan hoc la bat dau tu 1 => for 1 to size +1...kernel i-1 j-1 la do index matrix khac voi index math
        for j in range(size):
            temp = -float(((i - k) ** 2 + (j - k) ** 2)) / (2 * sigma ** 2)
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2) * np.exp(temp))

    ### END YOUR CODE
    # print(kernel)
    return kernel

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

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))
    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    G = zero_pad(G, 1, 1)
    # print(G)
    ### BEGIN YOUR CODE
    for x in range(H):
        for y in range(W):
            if theta[x][y] == 0:
                out[x, y] = G[x + 1, y + 1] if G[x + 1, y + 1] > G[x + 1, y + 1 + 1] and G[x + 1, y + 1] > G[
                    x + 1, y + 1 - 1] else 0
            if theta[x][y] == 45:
                out[x, y] = G[x + 1, y + 1] if G[x + 1, y + 1] > G[x + 1 + 1, y + 1 - 1] and G[x + 1, y + 1] > G[
                    x + 1 - 1, y + 1 + 1] else 0
            if theta[x][y] == 90:
                out[x, y] = G[x + 1, y + 1] if G[x + 1, y + 1] > G[x - 1 + 1, y + 1] and G[x + 1, y + 1] > G[
                    x + 1 + 1, y + 1] else 0
            if theta[x][y] == 135:
                out[x, y] = G[x + 1, y + 1] if G[x + 1, y + 1] > G[x + 1 - 1, y + 1 - 1] and G[x + 1, y + 1] > G[
                    x + 1 + 1, y + 1 + 1] else 0

    ### END YOUR CODE
    return out

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter
    Returns:
        response: Harris response image of shape (H, W)
    """
    H, W = img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    Ix2 = dx **2
    Iy2 = dy **2
    Ixy = dx * dy

    Sx2 = convolve(Ix2, window)
    Sy2 = convolve(Iy2, window)
    Sxy = convolve(Ixy, window)
    for i in range(H - 3):
        for j in range(W - 3):
            M = np.matrix([[Sx2[i, j], Sxy[i, j]], [Sxy[i, j], Sy2[i, j]]])
            eig, ez = np.linalg.eig(M)
            lamda1 = eig[0]
            lamda2 = eig[1]
            response[i, j] = (lamda1 * lamda2) - k * ((lamda1 + lamda2) ** 2)
    ### END YOUR CODE

    return response



def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []


    ### YOUR CODE HERE

    # for i in range(patch.shape[0]):
    #     for j in range(patch.shape[1]):
    #         feature.append(patch[i,j])
    #
    # feature=(feature-np.mean(feature))/np.linalg.norm((feature-np.mean(feature)))
    #

    mean_val = np.mean(patch)
    std_val = np.std(patch)
    if std_val==0:
        std_val=1
    for x in range(patch.shape[0]):
        for y in range(patch.shape[1]):
            feature.append((patch[x,y]-mean_val)/std_val)


    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    for x in range(len(dists)):
        temp_dists=np.copy(dists[x])
        sorted_dists_vec=np.sort(temp_dists)
        if (sorted_dists_vec[0]/sorted_dists_vec[1]) <threshold:
            matches.append([x,np.array(dists[x]).tolist().index(sorted_dists_vec[0])])



    # for x in range(dists.shape[0]):
    #     min1=np.min(dists[x])
    #     index_min1=np.array(dists[x]).tolist().index(min1)
    #     dists[x][index_min1]=999999
    #     min2=np.min(dists[x])
    #     index_min2 = np.array(dists[x]).tolist().index(min2)
    #     if min1/min2<threshold:
    #         matches.append([x,index_min1])

    matches=np.array(matches)
    print(len(matches))
    ### END YOUR CODE
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Hint:
        You can use np.linalg.lstsq function to solve the problem. 
        
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)
    ### YOUR CODE HERE
    H=np.linalg.lstsq(p2,p1)[0]

    # if p1.shape[0]<= p1.shape[1]:
    #     p2_inv= np.matrix(p2).T*np.linalg.inv(np.matrix(p2)*np.matrix(p2).T)
    # else :
    #     p2_inv= np.linalg.inv(np.matrix(p2).T*np.matrix(p2))*np.matrix(p2).T
    #
    # H=p2_inv*p1
    # H=np.matrix(H)
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]

    H[:,2] = np.array([0, 0, 1])
    return H

def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:, 0]])
    matched2 = pad(keypoints2[matches[:, 1]])

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    while  n_iters>0:  ## Lap 200
        # Buoc 1 :
        shuffle_idxs = np.arange(N)
        np.random.shuffle(shuffle_idxs)  # xao tron , lay ngau nhien

        idxs_sample = shuffle_idxs[:n_samples]
        idxs_test = shuffle_idxs[n_samples:]

        sample_matched1 = matched1[idxs_sample]
        sample_matched2 = matched2[idxs_sample]

        # Buoc 2 :

        h = np.linalg.lstsq(sample_matched2, sample_matched1)[0]
        h[:, 2] = np.array([0, 0, 1])

        test_matched1 = matched1[idxs_test]
        test_matched2 = matched2[idxs_test]

        # Buoc 3 :

        compute_test = np.dot(test_matched2, h)

        errors = np.sum((compute_test - test_matched1) ** 2, axis=1)

        alsoinlier_idxs = idxs_test[errors < threshold]

        current_inliners = len(alsoinlier_idxs) + n_samples
        ## Buoc 4 :
        if current_inliners > n_inliers:
            n_inliers = current_inliners
            max_inliers[idxs_sample] = True
            max_inliers[alsoinlier_idxs] = True
            inliners_matched1 = np.concatenate((sample_matched1, matched1[alsoinlier_idxs]))
            inliners_matched2 = np.concatenate((sample_matched2, matched2[alsoinlier_idxs]))
            H = np.linalg.lstsq(inliners_matched2, inliners_matched1)[0]
            H[:, 2] = np.array([0, 0, 1])
        ## Buoc 5 :
        n_iters -= 1
        ### END YOUR CODE
    return H, matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. compute the gradient image in x and y (already done for you)
    2. compute gradient histograms
    3. normalize across block 
    4. flattening block into a feature vector

    Args:
        patch: grayscale image patch of shape (h, w)
        pixels_per_cell: size of a cell with shape (m, n)

    Returns:
        block: 1D array of shape ((h*w*n_bins)/(m*n))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
   
    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    print(G_cells)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    pass
    ### YOUR CODE HERE
    
    return block
# matches.remove([362, 56])# cheo dai tren
#     matches.remove([404,304])#cheo dai duoi