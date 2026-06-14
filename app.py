from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import zipfile
import io
import os
import gc

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

TEMPLATE_PATH = 'watermark_template.png'

def process_image(file_bytes):
    try:
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
            
            # =======================================================
            # MAGIC FIX: Targeted ROI Search (Region of Interest)
            # =======================================================
            # যেহেতু ওয়াটারমার্কটি সবসময় নিচে ডানদিকে থাকে, তাই আমরা 
            # শুধুমাত্র নিচের ৪০% এবং ডানদিকের ৪০% জায়গায় স্ক্যান করবো।
            roi_y1 = int(h * 0.60)
            roi_x1 = int(w * 0.60)
            search_img = img[roi_y1:h, roi_x1:w]
            
            # যদি ছবি খুব ছোট হয়, তবে পুরো ছবিই স্ক্যান করবে
            if search_img.shape[0] < 50 or search_img.shape[1] < 50:
                roi_y1, roi_x1 = 0, 0
                search_img = img

            # স্কেল রেঞ্জ বাড়ানো হয়েছে (0.2 থেকে 3.0) যাতে যেকোনো সাইজের ছবিতে কাজ করে
            for scale in np.linspace(0.2, 3.0, 30):
                resized_w = int(template_bgr.shape[1] * scale)
                resized_h = int(template_bgr.shape[0] * scale)
                
                if resized_h > search_img.shape[0] or resized_w > search_img.shape[1] or resized_h == 0 or resized_w == 0: continue
                    
                res_bgr = cv2.resize(template_bgr, (resized_w, resized_h))
                res_alpha = cv2.resize(template_alpha, (resized_w, resized_h))
                
                # শুধুমাত্র ডানদিকের নিচের কোণায় (search_img) ম্যাচিং করা হচ্ছে
                res = cv2.matchTemplate(search_img, res_bgr, cv2.TM_CCORR_NORMED, mask=res_alpha)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if found is None or max_val > found[0]:
                    # মেইন ছবির সাথে লোকেশন ঠিক করার জন্য যোগ করা হচ্ছে
                    global_max_loc = (max_loc[0] + roi_x1, max_loc[1] + roi_y1)
                    found = (max_val, global_max_loc, scale, (resized_h, resized_w), res_alpha)
                    
            # নির্দিষ্ট এরিয়ায় খোঁজার কারণে থ্রেশহোল্ড 0.25 এ নামানো সম্ভব হয়েছে! 
            # এখন যত হিজিবিজি ব্যাকগ্রাউন্ডই হোক, সে ঠিকই খুঁজে বের করবে।
            threshold = 0.25
            if found and found[0] >= threshold:
                max_val, max_loc, scale, (th, tw), matched_alpha = found
                x1, y1 = max_loc
                
                _, tight_mask = cv2.threshold(matched_alpha, 15, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                tight_mask = cv2.dilate(tight_mask, kernel, iterations=1)
                
                main_mask = np.zeros((h, w), dtype=np.uint8)
                y2 = min(y1 + th, h)
                x2 = min(x1 + tw, w)
                mask_y_end = y2 - y1
                mask_x_end = x2 - x1
                
                main_mask[y1:y2, x1:x2] = tight_mask[0:mask_y_end, 0:mask_x_end]
                
                result = cv2.inpaint(img, main_mask, 3, cv2.INPAINT_TELEA)
                
        _, buffer = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return buffer.tobytes()
    except Exception as e:
        print(f"Error: {e}")
        return None

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
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        if files:
            for file in files:
                if file.filename == '': continue
                file_data = file.read()
                processed_bytes = process_image(file_data)
                if processed_bytes:
                    zf.writestr(f"clean_{file.filename}", processed_bytes)
                del file_data
                del processed_bytes
                gc.collect()
        
        if zip_file:
            with zipfile.ZipFile(zip_file, 'r') as uploaded_zip:
                for filename in uploaded_zip.namelist():
                    if filename.startswith('__MACOSX') or filename.endswith('/'): continue
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                    
                    file_data = uploaded_zip.read(filename)
                    processed_bytes = process_image(file_data)
                    
                    if processed_bytes:
                        zf.writestr(filename, processed_bytes)
                    
                    del file_data
                    del processed_bytes
                    gc.collect()

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='Cleaned_Images.zip')

@app.route('/', methods=['GET'])
def home():
    return "Server is Updated! Targeted ROI Search (Bottom-Right) is active."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
