from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/infer_frame")
async def infer_frame(file: UploadFile = File(...)):

    contents = await file.read()

    img = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    # ---- YOLO inference here ----
    # results = model(img)

    # For now just return same frame
    _, buffer = cv2.imencode(".jpg", img)

    encoded = base64.b64encode(buffer).decode()

    return {
        "device_used": "cpu",
        "max_conf": 0.9,
        "frame": encoded
    }



# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from aiortc import RTCPeerConnection, RTCSessionDescription
# from yolo_track import AnnotatedVideoTrack

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# pcs = set()

# @app.post("/infer_frame")
# async def offer(offer: dict):
#     pc = RTCPeerConnection()
#     pcs.add(pc)

#     @pc.on("track")
#     def on_track(track):
#         if track.kind == "video":
#             pc.addTrack(AnnotatedVideoTrack(track))

#     await pc.setRemoteDescription(
#         RTCSessionDescription(
#             sdp=offer["sdp"],
#             type=offer["type"]
#         )
#     )

#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     return {
#         "sdp": pc.localDescription.sdp,
#         "type": pc.localDescription.type
#     }
