import os
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEWEBSOCKET_COMPRESSION"] = "false"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import re
from rembg import remove
import io
import requests

st.set_page_config(page_title="Card OCR Scanner", layout="centered")
st.title("📇 Card OCR Scanner (Mobile Friendly)")

def format_date_ymd(ymd):
    if len(ymd) != 6:
        return "Invalid"
    return f"{ymd[4:6]}-{ymd[2:4]}-{ymd[0:2]}"

def parse_mrz(text):
    lines = [line.strip().replace(' ', '').replace('K', '<').replace('k', '<') for line in text.splitlines() if line.strip()]
    def is_mrz_line(line):
        return re.fullmatch(r'[A-Z0-9<]{20,}', line) is not None

    candidates = []
    for i in range(len(lines) - 2):
        l1, l2, l3 = lines[i:i+3]
        if all(is_mrz_line(line) for line in [l1, l2, l3]):
            candidates.append((l1, l2, l3))
    if not candidates:
        return {"error": "Valid MRZ block not found"}

    l1, l2, l3 = candidates[0]

    id_match = re.search(r'[A-Z]{3}([0-9]{8})<', l1)
    id_number = id_match.group(1) if id_match else None

    dob_raw = l2[0:6]
    gender = l2[7]
    expiry_raw = l2[8:14]

    dob = format_date_ymd(dob_raw)
    expiry = format_date_ymd(expiry_raw)

    name_parts = l3.split('<', 1)
    first_name = name_parts[0].replace('<', '')
    last_name = name_parts[1].replace('<', '') if len(name_parts) > 1 else ''

    return {
        "ID Number": id_number,
        "Date of Birth (YY-MM-DD)": dob,
        "Gender": gender,
        "Expiry Date (YY-MM-DD)": expiry,
        "First Name": first_name[:25],
        "Last Name": last_name[:25]
    }

def process_image(uploaded_file):
    # Load image
    input_image = Image.open(io.BytesIO(uploaded_file.read()))

    # Remove background
    output_image = remove(input_image)

    # Crop transparent edges
    np_img = np.array(output_image)
    if np_img.shape[2] == 4:
        alpha = np_img[:, :, 3]
        non_empty_cols = np.where(np.max(alpha, axis=0) > 0)[0]
        non_empty_rows = np.where(np.max(alpha, axis=1) > 0)[0]
        if non_empty_cols.size and non_empty_rows.size:
            crop_box = (non_empty_cols[0], non_empty_rows[0], non_empty_cols[-1], non_empty_rows[-1])
            cropped_image = output_image.crop(crop_box)
        else:
            cropped_image = output_image
    else:
        cropped_image = output_image

    # Resize for better OCR results
    w_percent = (600 / float(cropped_image.width))
    h_size = int((float(cropped_image.height) * float(w_percent)))
    resized = cropped_image.resize((600, h_size))

    # Prepare image bytes for OCR API
    buffered = io.BytesIO()
    resized.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Call OCR.Space API
    api_url = "https://api.ocr.space/parse/image"
    headers = {
        "apikey": "helloworld"  # public demo key, replace with your own key if you have one
    }
    files = {
        'filename': ('image.png', img_bytes)
    }
    data = {
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': 2,
        'detectOrientation': True
    }
    response = requests.post(api_url, headers=headers, files=files, data=data)
    result_json = response.json()

    # Extract OCR text
    try:
        text = result_json['ParsedResults'][0]['ParsedText']
    except Exception:
        text = ""

    # Fix common MRZ OCR issues (K → <)
    text = re.sub(r'(?<=[A-Z0-9])K(?=[A-Z0-9<])', '<', text)

    # Parse MRZ from text
    result = parse_mrz(text)

    return resized, text, result


uploaded_file = st.camera_input("Take a photo of your card")

if uploaded_file is not None:
    with st.spinner("Processing image with OCR API..."):
        processed_img, ocr_text, parsed_data = process_image(uploaded_file)

    st.image(processed_img, caption="Processed Image", use_column_width=True)

    st.subheader("Extracted MRZ Text:")
    st.text_area("OCR Text", ocr_text, height=200)

    st.subheader("Parsed Data:")
    st.json(parsed_data)
