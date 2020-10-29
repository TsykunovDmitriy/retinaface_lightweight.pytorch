import os
import cv2
import torch

from .utils.inference import *
from .utils.align import warp_and_crop_face
from .models.retinaface import RetinaFace, PriorBox
from .models.config import cfg_mnet as cfg

current_dir = os.path.dirname(os.path.abspath(__file__))

class RetinaDetector:
    def __init__(
                self, 
                device="cpu", 
                weights=f"{current_dir}/weights/mobilenet0.25_Final.pth", 
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

    @staticmethod
    def aligning(img, lands, crop_size=(512, 512)):
        return warp_and_crop_face(img, lands, crop_size=crop_size)

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

        boxes = dets[:, :4]
        scores = dets[:, 4]
        landms = landms.reshape(-1, 5, 2)

        return boxes, landms, scores
    











        

        

        



