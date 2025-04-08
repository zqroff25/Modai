# blip_api.py
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

@app.route("/blip", methods=["POST"])
def caption_image():
    if "file" not in request.files:
        return jsonify({"error": "Görsel yüklenmedi"}), 400

    image_file = request.files["file"]
    image = Image.open(image_file.stream).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return jsonify({"description": description})

if __name__ == "__main__":
    app.run(port=8600)
