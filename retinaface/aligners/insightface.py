import cv2
import numpy as np


def convert_to_5_landmarks(lm):
    """
    Convert Dlib (68 points) to InsightFace format (5 points)
    """

    eye_left = np.mean(lm[36:42], axis=0)  # Mean X and mean Y of left eye
    eye_right = np.mean(lm[42:48], axis=0)  # The same as above but for right eye
    nose = lm[30]
    mouth_left = lm[48]
    mouth_right = lm[54]

    return np.stack([eye_left, eye_right, nose, mouth_left, mouth_right], axis=0)


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T



REFERENCE_FACIAL_POINTS = np.array(
    [
        [0.34191607, 0.46157411],
        [0.65653392, 0.45983393],
        [0.500225, 0.64050538],
        [0.3709759, 0.82469198],
        [0.63151697, 0.82325091],
    ]
)


def insightface_align(image, landmarks, output_size=1024, *args, **kwargs):
    """
    Args:
        image (np.ndarray): Input image
        landmarks (list or np.ndarray): Facial landmarks (68 or 5 points)
        output_size (int, optional): Output face resolution. Defaults to 1024.

    Returns:
        np.ndarray: Image with aligned face
    """
    # convert to np.ndarray
    if isinstance(landmarks, list):
        lm = np.array(landmarks)

    # convert to 5 points
    if landmarks.shape[0] == 68:
        lm = convert_to_5_landmarks(lm)

    M = umeyama(lm, REFERENCE_FACIAL_POINTS * output_size, True)[:2]
    return cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        borderMode=cv2.BORDER_REFLECT,
        flags=cv2.INTER_CUBIC,
    )
