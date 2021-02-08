import cv2
import numpy as np


def boxfilter(I, r):
    """
    Apply box filter on the input image

    Params:
        I -- (narray) input image
        r -- (int) kernel size

    Returns:
        (narray)
    """
    bf = cv2.boxFilter(I, -1, (r, r))
    return bf



def guidefilter(I, P, r, eps):
    """
    This function applies guided filter to the input image P using a guide Image I.

    Params:
        I -- (narray) Guide image
        P -- (narray) Image to be filted
        r -- (int) Kernel size
        eps -- (float) Regularization coefficient

    Returns:
        (narray)
    """

    N = boxfilter(np.ones(np.shape(I)), r)  # size of each local patch

    mean_I = boxfilter(I, r) / N
    mean_P = boxfilter(P, r) / N
    mean_IP = boxfilter(I * P, r) / N
    cov_IP = mean_IP - mean_I * mean_P  # covariance of (I,P) in each local patch

    mean_II = boxfilter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_IP / (var_I + eps)  # equation 5 in the paper
    b = mean_P - a * mean_I  # equation 6 in the paper

    mean_a = boxfilter(a, r) / N
    mean_b = boxfilter(b, r) / N

    q = mean_a * I + mean_b  # equation 8 in the paper
    return q



def guidefilter_ite(img, r_start, eps_start, ite):
    """
    Apply guided filter in specified number of iterations, for each of which the kernel size increases while eps
    decreases.

    Params:
        img -- (narrray) Input data for filtering
        r_start -- (int) Kernel size in first iteration
        eps_start - (float) Regularization coefficient in first iteration
        ite -- (int) iteration

    Returns:
        (narray)
    """
    r = r_start
    eps = eps_start

    for i in range(ite):
        eps /= 3 ** i
        img = guidefilter(img, img, r, eps)
        r += 2

    return img


def guidefilter_ite2(img, r_start, eps_start, ite):

    for i in range(2):
        img = guidefilter_ite(img, r_start, eps_start, ite)

    return img