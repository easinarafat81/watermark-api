from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import zipfile
import io
import os

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

TEMPLATE_PATH = 'watermark_template.png'

def process_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None
    
    if not os.path.exists(TEMPLATE_PATH):
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return buffer.tobytes()
        
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    result = img.copy()
    
    if len(template.shape) == 3 and template.shape[2] == 4:
        template_bgr = template[:, :, :3]
        template_alpha = template[:, :, 3]
        
        found = None
        for scale in np.linspace(0.5, 1.5, 15):
            resized_w = int(template_bgr.shape[1] * scale)
            resized_h = int(template_bgr.shape[0] * scale)
            
            if resized_h > h or resized_w > w or resized_h == 0 or resized_w == 0: continue
                
            res_bgr = cv2.resize(template_bgr, (resized_w, resized_h))
            res_alpha = cv2.resize(template_alpha, (resized_w, resized_h))
            
            res = cv2.matchTemplate(img, res_bgr, cv2.TM_CCORR_NORMED, mask=res_alpha)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if found is None or max_val > found[0]:
                found = (max_val, max_loc, scale, (resized_h, resized_w), res_alpha)
                
        threshold = 0.65
        if found and found[0] >= threshold:
            max_val, max_loc, scale, (th, tw), matched_alpha = found
            x1, y1 = max_loc
            
            x2 = min(x1 + tw, w)
            y2 = min(y1 + th, h)
            actual_tw = x2 - x1
            actual_th = y2 - y1
            
            if actual_tw > 0 and actual_th > 0:
                roi = result[y1:y2, x1:x2].astype(np.float32)
                
                # শুধু তারার শেপটুকু নেওয়া হচ্ছে
                shape_mask = matched_alpha[0:actual_th, 0:actual_tw].astype(np.float32)
                
                # শেপের চারপাশ হালকা ব্লেন্ড করার জন্য Gaussian Blur (যাতে হার্ড লাইন না থাকে)
                shape_mask = cv2.GaussianBlur(shape_mask, (3, 3), 0)
                
                # ==========================================
                # ম্যাজিক সেকশন (ওয়াটারমার্কের সাদা রং বিয়োগ করা)
                # ==========================================
                # ওয়াটারমার্কের অপাসিটি বা গাঢ়ত্ব (ডিফল্ট 0.35 বা 35% রাখা হয়েছে)
                OPACITY_LEVEL = 0.35 
                
                normalized_alpha = (shape_mask / 255.0) * OPACITY_LEVEL
                alpha_3d = np.repeat(normalized_alpha[:, :, np.newaxis], 3, axis=2)
                alpha_3d = np.clip(alpha_3d, 0, 0.95) # এরর ঠেকানোর জন্য
                
                # গাণিতিক সূত্র: Original Pixel = (Watermarked Pixel - White * Alpha) / (1 - Alpha)
                # এই সূত্রটি ব্যাকগ্রাউন্ড ব্লার না করে শুধুমাত্র সাদা প্রলেপটি তুলে নেবে!
                recovered = (roi - 255.0 * alpha_3d) / (1.0 - alpha_3d)
                
                recovered = np.clip(recovered, 0, 255).astype(np.uint8)
                result[y1:y2, x1:x2] = recovered
                
    _, buffer = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return buffer.tobytes()

@app.route('/process', methods=['POST'])
def process():
    files = request.files.getlist('images')
    zip_file = request.files.get('zipfile')

    if files and len(files) == 1 and files[0].filename != '' and not zip_file:
        file = files[0]
        processed_bytes = process_image(file.read())
        if processed_bytes:
            return send_file(io.BytesIO(processed_bytes), mimetype='image/jpeg', as_attachment=True, download_name=f"clean_{file.filename}")

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        if files:
            for file in files:
                if file.filename == '': continue
                processed_bytes = process_image(file.read())
                if processed_bytes:
                    zf.writestr(f"clean_{file.filename}", processed_bytes)
        
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as uploaded_zip:
                for filename in uploaded_zip.namelist():
                    if filename.startswith('__MACOSX') or filename.endswith('/'): continue
                    file_data = uploaded_zip.read(filename)
                    processed_bytes = process_image(file_data)
                    if processed_bytes:
                        zf.writestr(filename, processed_bytes)

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='Cleaned_Images.zip')

@app.route('/', methods=['GET'])
def home():
    return "Server is Live! Inpainting removed, using Math Subtraction."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
