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
                
        # থ্রেশহোল্ড 0.65 রাখা হলো যাতে সহজেই ওয়াটারমার্ক খুঁজে পায়
        threshold = 0.65
        if found and found[0] >= threshold:
            max_val, max_loc, scale, (th, tw), matched_alpha = found
            x1, y1 = max_loc
            
            main_mask = np.zeros((h, w), dtype=np.uint8)
            
            # --- ম্যাজিক ফিক্স ---
            # থ্রেশহোল্ড 100 করে দেওয়া হয়েছে। এতে remove.bg এর ফেলে যাওয়া চারকোনা অদৃশ্য বক্সটি আর ধরবে না।
            # শুধুমাত্র তারার সলিড শেপটুকুই মাস্ক হিসেবে কাজ করবে।
            _, exact_shape_mask = cv2.threshold(matched_alpha, 100, 255, cv2.THRESH_BINARY)
            
            y2 = min(y1 + th, h)
            x2 = min(x1 + tw, w)
            mask_y_end = y2 - y1
            mask_x_end = x2 - x1
            
            main_mask[y1:y2, x1:x2] = exact_shape_mask[0:mask_y_end, 0:mask_x_end]
            
            # আগের মতো মাস্ককে ডাইলেট (Dilate) করা হচ্ছে না, যাতে ঘাসের ওপর ব্লার না ছড়ায়।
            
            # INPAINT_TELEA এলগরিদম ছোট দাগ মোছার জন্য সেরা, রেডিয়াস 2 রাখা হয়েছে ব্যাকগ্রাউন্ড ঠিক রাখার জন্য।
            result = cv2.inpaint(img, main_mask, 2, cv2.INPAINT_TELEA)
            
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
    return "Server is Live and working with Precise Masking!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
