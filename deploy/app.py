"""Flask demo for the CycleGAN face <-> sketch model.

Run:  pip install -r requirements.txt && python app.py   ->  http://localhost:5000
"""
import base64, io, os

import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from cyclegan_core import ResnetGenerator, image_features, pil_to_tensor, tensor_to_pil

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------- load models once at start-up
ckpt = torch.load(os.path.join(BASE, 'generators.pth'), map_location=device)
IMG_SIZE = ckpt['img_size']
G_AB = ResnetGenerator(n_blocks=ckpt['n_res_blocks']).to(device)   # photo  -> sketch
G_BA = ResnetGenerator(n_blocks=ckpt['n_res_blocks']).to(device)   # sketch -> photo
G_AB.load_state_dict(ckpt['G_AB']); G_BA.load_state_dict(ckpt['G_BA'])
G_AB.eval(); G_BA.eval()

det_path = os.path.join(BASE, 'detector.joblib')
detector = joblib.load(det_path) if os.path.exists(det_path) else None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB upload limit


def detect_domain(img):
    """Return ('sketch' | 'photo', confidence). Uses the trained detector, or a simple saturation rule as fallback."""
    f = image_features(img)
    if detector is not None:
        p_sketch = float(detector.predict_proba(f[None])[0, 1])
    else:
        p_sketch = 1.0 if f[0] < 0.08 else 0.0          # sketches are (almost) colourless
    return ('sketch', p_sketch) if p_sketch >= 0.5 else ('photo', 1 - p_sketch)


def to_data_url(pil_img):
    """PIL image -> base64 PNG data URL for the browser."""
    buf = io.BytesIO(); pil_img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/convert', methods=['POST'])
def convert():
    """Accepts multipart form data: image (file) and optional mode = auto | photo | sketch."""
    if 'image' not in request.files:
        return jsonify(error='No image received.'), 400
    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
    except Exception:
        return jsonify(error='Could not read the file as an image.'), 400

    mode = request.form.get('mode', 'auto')
    if mode in ('photo', 'sketch'):
        domain, conf = mode, 1.0                         # manual override
    else:
        domain, conf = detect_domain(img)

    x = pil_to_tensor(img, IMG_SIZE).to(device)
    with torch.no_grad():
        y = G_BA(x) if domain == 'sketch' else G_AB(x)   # sketch -> face  |  face -> sketch
    return jsonify(detected=domain, confidence=round(conf, 3),
                   direction='sketch -> face' if domain == 'sketch' else 'face -> sketch',
                   input=to_data_url(img.resize((IMG_SIZE, IMG_SIZE))), output=to_data_url(tensor_to_pil(y)))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
