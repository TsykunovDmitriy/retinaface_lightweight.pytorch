# RetinaFace-lightweight

This repository contains script for inference RetinaFace with MobileNet encoder. The remake of the https://github.com/biubug6/Pytorch_Retinaface

### Installation

Clone this repository
```
pip3 install .
```
Make sure your CUDA version matches the latest PyTorch version. Otherwise, install the required version of PyTorch from [here](https://pytorch.org).

### How to use

Inference
```python
>>> import cv2
>>> from retinaface import RetinaDetector

>>> detector = RetinaDetector(
            device="cpu", # or number of GPU (example device=0)
            score_thresh=0.5, 
            top_k=100,
            nms_thresh=0.4
        )

>>> image = cv2.imread(path_to_image)
>>> bboxes, landmarks, scores = detector(image)
>>> print(bboxes, landmarks, scores)
(
    [
        [xmin, ymin, xmax, ymax] 
    ],
    [
        [[x, y], # right eye
         [x, y], # left eye
         [x, y], # nose
         [x, y], # right edge of the mouth
         [x, y]] # left edge of the mouth 
    ],
    [score]
)
```

Align
```python
>>> aligned_face, trm, trm_inv = detector.aligning(image, landmarks[person])
```
