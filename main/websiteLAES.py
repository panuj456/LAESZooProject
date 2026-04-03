'''
Tested and working with docker desktop
Easier deployment and cross-platform testing/development
'''

import base64
from io import BytesIO

import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import models, transforms

app = FastAPI()
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom trained model
path = "models/animal_model.pth" #works with model 1 not v2 - integrate your new model
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

@app.get("/")
async def home(request: Request):
    # Use context= specifically to avoid the 'unhashable type' error
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={}  # Put any extra variables here
    )

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
