import numpy as np

class Tracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id = 0

    def iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for track in self.tracks:
                iou_score = self.iou(det, track[:4])
                if iou_score > self.iou_threshold:
                    updated_tracks.append([*det, track[4]])
                    matched = True
                    break

            if not matched:
                updated_tracks.append([*det, self.track_id])
                self.track_id += 1

        self.tracks = updated_tracks
        return updated_tracks
