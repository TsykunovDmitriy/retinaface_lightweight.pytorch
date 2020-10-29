import cv2
from tqdm import tqdm

from retinaface import RetinaDetector
from retinaface.utils import video_utils

detector = RetinaDetector()

video = video_utils.vidread("./test_video.mp4")
output = []
for frame in tqdm(video):
    boxes, landms, scores = detector(frame)

    for i in range(len(boxes)):
        text = "{:.4f}".format(scores[i])
        cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 0, 255), 2)
        cx = boxes[i][0]
        cy = boxes[i][1] + 12
        cv2.putText(frame, text, (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(frame, (landms[i][0][0], landms[i][0][1]), 1, (0, 0, 255), 4)
        cv2.circle(frame, (landms[i][1][0], landms[i][1][1]), 1, (0, 255, 255), 4)
        cv2.circle(frame, (landms[i][2][0], landms[i][2][1]), 1, (255, 0, 255), 4)
        cv2.circle(frame, (landms[i][3][0], landms[i][3][1]), 1, (0, 255, 0), 4)
        cv2.circle(frame, (landms[i][4][0], landms[i][4][1]), 1, (255, 0, 0), 4)
    
    output.append(frame)

video_utils.vidwrite("output.mp4", output)