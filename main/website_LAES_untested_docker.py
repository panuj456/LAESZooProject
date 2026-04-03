'''
This code hasnt been tested but makes sure that code from latest commit from
Calpoly side is not lost through a merge
'''


import base64
from io import BytesIO

import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom trained model
path = '../models/animal_model_v2.pth'
checkpoint = torch.load(path, map_location=device)
categories = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(categories))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# Match your training preprocessing as closely as possible
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

HTML = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Animal Vision Camera</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 0;
      padding: 16px;
      background: #111;
      color: white;
    }
    #video, #canvas {
      width: 100%;
      max-width: 520px;
      border-radius: 12px;
      display: block;
      margin-bottom: 12px;
    }
    button {
      font-size: 16px;
      padding: 12px 16px;
      margin: 8px 8px 8px 0;
    }
    #result {
      margin-top: 12px;
      font-size: 18px;
    }
    #status {
      opacity: 0.8;
      margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <h2>Live Camera → Animal Vision Filter</h2>
  <div id="status">Not connected</div>
  <video id="video" autoplay playsinline muted style="display:none;"></video>
  <canvas id="canvas"></canvas>

  <div>
    <button id="startBtn">Start Camera</button>
    <button id="streamBtn">Start Streaming</button>
    <button id="stopBtn">Stop Streaming</button>
  </div>

  <div id="result">Prediction: waiting...</div>

  <script>
    const labelToFilter = {
      "alligator_sinensis": "reptile",
      "aonyx_cinereus": "mammal",
      "cacatua_galerita": "bird",
      "giraffa_camelopardalis_tippelskirchi": "mammal",
      "gorilla_gorilla_gorilla": "primate",
      "gymnogyps_californianus": "bird",
      "hydrochoerus_hydrochaeris": "mammal",
      "hylobates_lar": "primate",
      "macropus_fuliginosus": "mammal",
      "notamacropus_rufogriseus": "mammal",
      "panthera_leo": "bigcat",
      "panthera_pardus_orientalis": "bigcat",
      "panthera_uncia": "bigcat",
      "phoenicopterus_chilensis": "bird",
      "podargus_strigoides": "bird",
      "urocyon_littoralis": "canid"
    };
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");

    let ws = null;
    let stream = null;
    let intervalId = null;
    let currentFilter = "normal";
    let lastLabel = "none";

    async function startCamera() {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          statusEl.textContent = "Camera error: HTTPS required for camera access";
          return false;
        }

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 640 },
            height: { ideal: 480 }
          },
          audio: false
        });

        video.srcObject = stream;
        await video.play();
        statusEl.textContent = "Camera ready";
        return true;
      } catch (err) {
        statusEl.textContent = "Camera error: " + err.message;
        return false;
      }
    }

    function connectWS() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        statusEl.textContent = "WebSocket connected";
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.label) {
            resultEl.textContent = `Prediction: ${data.label} (${data.confidence.toFixed(1)}%)`;
            lastLabel = data.label.toLowerCase();
            currentFilter = labelToFilter[lastLabel] || "normal";
          } else if (data.error) {
            resultEl.textContent = "Error: " + data.error;
          }
        } catch (err) {
          resultEl.textContent = "Parse error: " + err.message;
        }
      };

      ws.onclose = () => {
        statusEl.textContent = "WebSocket closed";
      };
    }

function applyFilter(imageData, filterName) {
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    if (filterName === "bigcat") {
      // lower saturation, slightly brighter shadows
      const gray = 0.3 * r + 0.59 * g + 0.11 * b;
      data[i]     = Math.min(255, 0.55 * gray + 0.45 * r + 8);
      data[i + 1] = Math.min(255, 0.55 * gray + 0.45 * g + 8);
      data[i + 2] = Math.min(255, 0.55 * gray + 0.45 * b + 8);

    } else if (filterName === "canid") {
      // dog-like yellow/blue approximation
      const newR = 0.45 * r + 0.35 * g;
      const newG = 0.55 * g + 0.25 * b;
      const newB = 0.95 * b + 0.10 * g;
      data[i]     = Math.min(255, newR);
      data[i + 1] = Math.min(255, newG);
      data[i + 2] = Math.min(255, newB);

    } else if (filterName === "bird") {
      // more vivid / higher contrast artistic look
      data[i]     = Math.min(255, 1.15 * r);
      data[i + 1] = Math.min(255, 1.15 * g);
      data[i + 2] = Math.min(255, 1.2 * b);

    } else if (filterName === "reptile") {
      // heatmap-ish artistic reptile view
      const intensity = (r + g + b) / 3;
      data[i]     = intensity > 170 ? 255 : intensity;
      data[i + 1] = intensity > 100 ? 140 : 20;
      data[i + 2] = intensity < 90 ? 255 : 0;

    } else if (filterName === "primate") {
      // close to normal human-like color
      data[i]     = r;
      data[i + 1] = g;
      data[i + 2] = b;

    } else if (filterName === "mammal") {
      // mild desaturation
      const gray = 0.3 * r + 0.59 * g + 0.11 * b;
      data[i]     = 0.75 * r + 0.25 * gray;
      data[i + 1] = 0.75 * g + 0.25 * gray;
      data[i + 2] = 0.75 * b + 0.25 * gray;
    }
  }

  return imageData;
}

    function renderLoop() {
      if (video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        let frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
        frame = applyFilter(frame, currentFilter);
        ctx.putImageData(frame, 0, 0);
      }

      requestAnimationFrame(renderLoop);
    }

    function sendFrame() {
      if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;

      const tempCanvas = document.createElement("canvas");
      const tempCtx = tempCanvas.getContext("2d");

      const targetWidth = 224;
      const scale = targetWidth / video.videoWidth;
      const targetHeight = Math.round(video.videoHeight * scale);

      tempCanvas.width = targetWidth;
      tempCanvas.height = targetHeight;
      tempCtx.drawImage(video, 0, 0, targetWidth, targetHeight);

      const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.6);
      ws.send(dataUrl);
    }

    document.getElementById("startBtn").onclick = async () => {
      const ok = await startCamera();
      if (ok && (!ws || ws.readyState !== WebSocket.OPEN)) {
        connectWS();
      }
    };

    document.getElementById("streamBtn").onclick = async () => {
      let ok = true;

      if (!stream) {
        ok = await startCamera();
      }

      if (!ok) {
        statusEl.textContent = "Cannot stream: camera not available";
        return;
      }

      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWS();
      }

      if (intervalId) clearInterval(intervalId);
      intervalId = setInterval(sendFrame, 400);

      statusEl.textContent = "Streaming frames...";
    };

    document.getElementById("stopBtn").onclick = () => {
      if (intervalId) clearInterval(intervalId);
      intervalId = null;
      statusEl.textContent = "Streaming stopped";
    };

    renderLoop();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_url = await websocket.receive_text()

            if "," not in data_url:
                await websocket.send_json({"error": "Bad frame format"})
                continue

            _, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            x = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.nn.functional.softmax(logits[0], dim=0)
                top_prob, top_idx = torch.max(probs, dim=0)

            label = categories[top_idx.item()]
            confidence = float(top_prob.item()) * 100.0

            await websocket.send_json({"label": label, "confidence": confidence})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass