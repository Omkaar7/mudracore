from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import torch
import logging
from ultralytics import YOLO

# =========================
# LOGGING SETUP
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mudra-server")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="KathakMudra YOLO Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBALS
# =========================
model = None
device = "cpu"

# =========================
# STARTUP EVENT
# =========================
@app.on_event("startup")
def startup_event():
    global model, device

    logger.info("🚀 Starting KathakMudra Inference Server")
    logger.info(f"🧠 Torch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    logger.info(f"🟢 CUDA available: {cuda_available}")

    if cuda_available:
        logger.info(f"🎮 CUDA version: {torch.version.cuda}")
        logger.info(f"📊 GPU count: {torch.cuda.device_count()}")

        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🔥 Using device: {device}")
        logger.info(f"🎯 GPU name: {gpu_name}")
    else:
        device = "cpu"
        logger.warning("⚠️ CUDA not available — using CPU")

    # Load YOLO model
    logger.info("📦 Loading YOLO model...")
    model = YOLO("best.pt")
    model.to(device)

    # Warm-up (important for GPU)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)

    logger.info("✅ YOLO model loaded and warmed up successfully")

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }

@app.post("/infer_frame")
def infer_frame(file: UploadFile = File(...)):
    image_bytes = file.file.read()

    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)
    encoded = base64.b64encode(buffer).decode("utf-8")

    max_conf = (
        float(results[0].boxes.conf.max())
        if results[0].boxes is not None and len(results[0].boxes) > 0
        else 0.0
    )

    return {
        "device_used": device,
        "max_conf": max_conf,
        "frame": encoded,
    }




# # server.py
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import base64
# from ultralytics import YOLO

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = YOLO("phase1.pt")
# model.to("cuda:0")


# @app.get("/")
# def health():
#     return {"status": "ok"}

# @app.post("/infer_frame")
# async def infer_frame(file: UploadFile = File(...)):
#     image_bytes = await file.read()

#     np_img = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     results = model(frame, conf=0.5, verbose=False)
#     annotated = results[0].plot()

#     _, buffer = cv2.imencode(".jpg", annotated)
#     encoded = base64.b64encode(buffer).decode("utf-8")

#     return {
#         "frame": encoded,
#         "max_conf": float(results[0].boxes.conf.max()) if results[0].boxes else 0.0
#     }


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
