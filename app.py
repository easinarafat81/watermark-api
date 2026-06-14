from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import zipfile
import io
import os
import gc
from PIL import Image, PngImagePlugin  # মেটাডেটা হ্যান্ডেল করার জন্য নতুন যুক্ত করা হলো

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])

TEMPLATE_PATH = 'watermark_template.png'

def process_image(file_bytes, filename):
    try:
        # ১. OpenCV দিয়ে ইমেজ প্রসেসিং
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None
        
        ext = os.path.splitext(filename)[1].lower()
        
        if not os.path.exists(TEMPLATE_PATH):
            return file_bytes # অরিজিনাল ফাইলই ফেরত দিয়ে দিলাম
            
        template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        result = img.copy()
        
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_bgr = template[:, :, :3]
            template_alpha = template[:, :, 3]
            
            found = None
            roi_y1 = int(h * 0.60)
            roi_x1 = int(w * 0.60)
            search_img = img[roi_y1:h, roi_x1:w]
            
            if search_img.shape[0] < 50 or search_img.shape[1] < 50:
                roi_y1, roi_x1 = 0, 0
                search_img = img

            for scale in np.linspace(0.2, 3.0, 30):
                resized_w = int(template_bgr.shape[1] * scale)
                resized_h = int(template_bgr.shape[0] * scale)
                
                if resized_h > search_img.shape[0] or resized_w > search_img.shape[1] or resized_h == 0 or resized_w == 0: continue
                    
                res_bgr = cv2.resize(template_bgr, (resized_w, resized_h))
                res_alpha = cv2.resize(template_alpha, (resized_w, resized_h))
                
                res = cv2.matchTemplate(search_img, res_bgr, cv2.TM_CCORR_NORMED, mask=res_alpha)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if found is None or max_val > found[0]:
                    global_max_loc = (max_loc[0] + roi_x1, max_loc[1] + roi_y1)
                    found = (max_val, global_max_loc, scale, (resized_h, resized_w), res_alpha)
                    
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
                
        # ==========================================
        # ২. PIL দিয়ে মেটাডেটা রিকভারি এবং ইনজেকশন
        # ==========================================
        # OpenCV BGR-এ কাজ করে, PIL RGB-তে। তাই কালার চ্যানেল কনভার্ট করা হচ্ছে
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        
        # অরিজিনাল ছবিটি মেমরিতে রিড করে তার অদৃশ্য মেটাডেটা বের করে আনা হচ্ছে
        orig_pil = Image.open(io.BytesIO(file_bytes))
        icc_profile = orig_pil.info.get('icc_profile') # কালার কোয়ালিটি প্রোফাইল
        
        out_bytes = io.BytesIO()
        
        if ext == '.png':
            # PNG ছবির জন্য টেক্সট চ্যাঙ্ক (Text Chunks) রিকভার করা
            pnginfo = PngImagePlugin.PngInfo()
            for k, v in orig_pil.info.items():
                if isinstance(v, str):
                    pnginfo.add_text(k, v)
            
            # মেটাডেটা সহ সেভ করা
            pil_img.save(out_bytes, format='PNG', pnginfo=pnginfo, icc_profile=icc_profile)
        else:
            # JPG ছবির জন্য EXIF Data (ক্যামেরা, লোকেশন, তারিখ) রিকভার করা
            exif = orig_pil.info.get('exif')
            save_kwargs = {'format': 'JPEG', 'quality': 100}
            
            if exif: 
                save_kwargs['exif'] = exif
            if icc_profile: 
                save_kwargs['icc_profile'] = icc_profile
                
            pil_img.save(out_bytes, **save_kwargs)
            
        return out_bytes.getvalue()
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route('/process', methods=['POST'])
def process():
    files = request.files.getlist('images')
    zip_file = request.files.get('zipfile')

    if files and len(files) == 1 and files[0].filename != '' and not zip_file:
        file = files[0]
        processed_bytes = process_image(file.read(), file.filename)
        if processed_bytes:
            mimetype = 'image/png' if file.filename.lower().endswith('.png') else 'image/jpeg'
            return send_file(io.BytesIO(processed_bytes), mimetype=mimetype, as_attachment=True, download_name=f"clean_{file.filename}")

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        if files:
            for file in files:
                if file.filename == '': continue
                file_data = file.read()
                processed_bytes = process_image(file_data, file.filename)
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
                    processed_bytes = process_image(file_data, filename)
                    
                    if processed_bytes:
                        zf.writestr(filename, processed_bytes)
                    
                    del file_data
                    del processed_bytes
                    gc.collect()

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='Cleaned_Images.zip')

@app.route('/', methods=['GET'])
def home():
    return "Server is Updated! Metadata (EXIF) and ICC Profiles are fully preserved."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
