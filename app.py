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
        
    # টেমপ্লেটটি আলফা চ্যানেল (Transparency) সহ রিড করা
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
    
    h, w = img.shape[:2]
    result = img.copy()
    
    # টেমপ্লেটে ট্রান্সপারেন্ট ব্যাকগ্রাউন্ড আছে কিনা চেক করা
    if len(template.shape) == 3 and template.shape[2] == 4:
        template_bgr = template[:, :, :3]
        template_alpha = template[:, :, 3]
        
        found = None
        # স্কেলিং করে সঠিক সাইজটি খোঁজা
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
                
        threshold = 0.75
        if found and found[0] >= threshold:
            max_val, max_loc, scale, (th, tw), matched_alpha = found
            x1, y1 = max_loc
            
            # ১. পুরো ছবির সাইজের একটি ফাঁকা কালো মাস্ক তৈরি
            main_mask = np.zeros((h, w), dtype=np.uint8)
            
            # ২. টেমপ্লেটের আলফা চ্যানেল থেকে শুধু তারার শেপ (Shape) বের করা
            _, exact_shape_mask = cv2.threshold(matched_alpha, 10, 255, cv2.THRESH_BINARY)
            
            # ৩. মাস্কটি ছবির সীমানার বাইরে যেন না যায় তার হিসাব
            y2 = min(y1 + th, h)
            x2 = min(x1 + tw, w)
            mask_y_end = y2 - y1
            mask_x_end = x2 - x1
            
            # ৪. স্কয়ার বক্সের বদলে একদম নিখুঁত শেপটি মূল মাস্কে বসিয়ে দেওয়া
            main_mask[y1:y2, x1:x2] = exact_shape_mask[0:mask_y_end, 0:mask_x_end]
            
            # ৫. শেপটিকে সুক্ষ্মভাবে একটু মোটা করা (যাতে ওয়াটারমার্কের চারপাশের হালকা সাদাও কভার হয়)
            kernel = np.ones((3, 3), np.uint8)
            main_mask = cv2.dilate(main_mask, kernel, iterations=1)
            
            # ৬. ইনপেইন্টিং (INPAINT_NS টেক্সচার রিকভারির জন্য ভালো কাজ করে এবং এরিয়া ছোট করা হয়েছে)
            result = cv2.inpaint(img, main_mask, 2, cv2.INPAINT_NS)
            
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
    return "Server is Live and Running Perfectly!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
