from aiortc import VideoStreamTrack
from av import VideoFrame
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

class AnnotatedVideoTrack(VideoStreamTrack):
    def __init__(self, source_track):
        super().__init__()
        self.track = source_track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=0.5, verbose=False)
        annotated = results[0].plot()

        new_frame = VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame
