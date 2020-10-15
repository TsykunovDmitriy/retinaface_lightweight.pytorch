import cv2
import torch

from utils.inference import *
from models.retinaface import RetinaFace, PriorBox
from models.config import cfg_mnet as cfg

class RetinaDetector:
    def __init__(
                self, 
                device="cpu", 
                weights="./weights/mobilenet0.25_Final.pth", 
                score_thresh=0.5, 
                top_k=100,
                nms_thresh=0.4,
            ):
        try:
            device = int(device)
            device = f"cuda:{device}"
        except ValueError:
            pass
        self.device = torch.device(device)

        self.detector = RetinaFace(cfg)
        load_model(self.detector, weights, self.device)
        self.detector.eval()
        self.detector.to(self.device)
        
        self.priorbox = PriorBox(cfg).forward().to(self.device)

        self.score_thresh = score_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def __call__(self, image):
        image_ = pad(image)
        scale = cfg["image_size"] / image_.shape[0]
        image_ = cv2.resize(image_, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        image_ = image_.astype(np.float32)
        image_ -= (104, 117, 123)
        image_ = image_.transpose(2, 0, 1)
        image_ = torch.from_numpy(image_).unsqueeze(0)
        image_ = image_.to(self.device)

        with torch.no_grad():
            loc, conf, landms = self.detector(image_)

        boxes = decode(loc.data.squeeze(0), self.priorbox, cfg['variance'])
        boxes = boxes * image_.size(2) / scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), self.priorbox, cfg['variance'])
        landms = landms * image_.size(2) / scale
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.score_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        return dets

if __name__ == "__main__":
    from utils import video_utils
    from tqdm import tqdm
    from filters.kernel_filters import median

    detector = RetinaDetector()
    
    video, audio_path, fps = video_utils.vidread("./test_video.mp4")
    landmarks = []
    boxes = []
    scores = []
    idx_not_empty = []
    for i, frame in tqdm(enumerate(video)):
        dets = detector(frame)
        if len(dets):
            idx_not_empty.append(i)
            boxes.append(dets[0][:4])
            scores.append(dets[0][4])
            landmarks.append(dets[0][5:].reshape(5, 2))
        # if i == 100:
        #     break

    land = median(np.array(landmarks), m=5).astype(np.int)
    boxes = np.array(boxes)
    scores = np.array(scores)


    output = []
    for i, frame in enumerate(video):
        if i in idx_not_empty:
            text = "{:.4f}".format(scores[0])
            cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (0, 0, 255), 2)
            cx = boxes[0][0]
            cy = boxes[0][1] + 12
            cv2.putText(frame, text, (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.circle(frame, (land[0][0][0], land[0][0][1]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (land[0][1][0], land[0][1][1]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (land[0][2][0], land[0][2][1]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (land[0][3][0], land[0][3][1]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (land[0][4][0], land[0][4][1]), 1, (255, 0, 0), 4)
            land = np.delete(land, 0, 0)
            boxes = np.delete(boxes, 0, 0)
            scores = np.delete(scores, 0, 0)
        output.append(frame[..., ::-1])

    #     for b in dets:
    #         text = "{:.4f}".format(b[4])
    #         cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    #         cx = b[0]
    #         cy = b[1] + 12
    #         cv2.putText(frame, text, (int(cx), int(cy)),
    #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    #         landmarks = b[5:].reshape(5, 2)
    #         opt_landmarks = []
    #         for i in range(5):
    #             kalmans[i].update(landmarks[i])
    #             opt_landmarks.append(kalmans[i].get_results())

    #         land = opt_landmarks
    #         # landms
    #         cv2.circle(frame, (land[0][0], land[0][1]), 1, (0, 0, 255), 4)
    #         cv2.circle(frame, (land[1][0], land[1][1]), 1, (0, 255, 255), 4)
    #         cv2.circle(frame, (land[2][0], land[2][1]), 1, (255, 0, 255), 4)
    #         cv2.circle(frame, (land[3][0], land[3][1]), 1, (0, 255, 0), 4)
    #         cv2.circle(frame, (land[4][0], land[4][1]), 1, (255, 0, 0), 4)
        
    #     output.append(frame[..., ::-1])

    out_video = video_utils.vidwrite(output, audio_path, fps)
    video_utils.write_bytesio_to_file("./test_video_out.mp4", out_video)
    











        

        

        




