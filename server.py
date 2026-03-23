from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription
from yolo_track import AnnotatedVideoTrack

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs = set()

@app.post("/offer")
async def offer(offer: dict):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(AnnotatedVideoTrack(track))

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=offer["sdp"],
            type=offer["type"]
        )
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
