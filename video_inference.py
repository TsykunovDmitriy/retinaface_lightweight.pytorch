import cv2
import argparse
import numpy as np
from tqdm import tqdm

from retinaface import RetinaDetector
from retinaface.utils import video_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="path to video")
    parser.add_argument("--output", type=str, help="path to save")
    parser.add_argument("--device", default="cpu", help="device for inference")
    parser.add_argument("--use_trt", action='store_true', help="if true use tensorrt for inference")
    args = parser.parse_args()

    detector = RetinaDetector(device=args.device, use_trt=args.use_trt)

    video = video_utils.vidread(args.video)
    output = []
    for frame in tqdm(video):
        boxes, landms, scores = detector(frame)
        boxes = boxes.astype(np.int)
        landms = landms.astype(np.int)

        for i in range(len(boxes)):
            text = "{:.4f}".format(scores[i])
            cv2.rectangle(frame, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 2)
            cx = boxes[i][0]
            cy = boxes[i][1] + 12
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(frame, (landms[i][0][0], landms[i][0][1]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (landms[i][1][0], landms[i][1][1]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (landms[i][2][0], landms[i][2][1]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (landms[i][3][0], landms[i][3][1]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (landms[i][4][0], landms[i][4][1]), 1, (255, 0, 0), 4)
        
        output.append(frame)

    video_utils.vidwrite(args.output, output)
    