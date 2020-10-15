import cv2
import numpy as np
from scipy import signal

def average(inp, m=3):
    """
    Apply average filter to input.

    Args:
        input: numpy.array [num x 5 x 2]
        m: int - size of kernel

    Returns:
        numpy.array [num x 5 x 2]
    """
    assert m % 2 == 1, "Kernel size should be odd"
    weights = np.zeros((m, m), np.float32)
    weights[m // 2, :] = 1
    weights /= m
    return cv2.filter2D(inp.transpose(1, 0, 2), -1, weights).transpose(1, 0, 2)

def median(inp, m=3):
    """
    Apply median filter to input.

    Args:
        input: numpy.array [num x 5 x 2]
        m: int - size of kernel

    Returns:
        numpy.array [num x 5 x 2]
    """
    assert m % 2 == 1, "Kernel size should be odd"
    opt_landmarks = []
    for i in range(5):
        x = signal.medfilt(inp[:, i, 0], m).reshape(-1, 1)
        y = signal.medfilt(inp[:, i, 1], m).reshape(-1, 1)
        opt_landmarks.append(np.concatenate((x, y), axis=1))
    return np.array(opt_landmarks).transpose(1, 0, 2)