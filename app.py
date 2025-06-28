import os
import io
import cv2
import re
import json
import numpy as np
from PIL import Image
import streamlit as st
from rembg import remove
from google.cloud import vision
from google.oauth2 import service_account

# Streamlit config
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEWEBSOCKET_COMPRESSION"] = "false"
st.set_page_config(page_title="Card OCR Scanner", layout="centered")
st.title("📇 Card OCR Scanner (Mobile Friendly)")

# Google Vision API setup
creds = service_account.Credentials.from_service_account_info(
    st.secrets["google_credentials"]
)
client = vision.ImageAnnotatorClient(credentials=creds)

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

def run_ocr_google(image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    text = response.full_text_annotation.text
    return text

def process_image(uploaded_file):
    input_image = Image.open(io.BytesIO(uploaded_file.read()))
    output_image = remove(input_image)

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

    w_percent = (600 / float(cropped_image.width))
    h_size = int((float(cropped_image.height) * float(w_percent)))
    resized = cropped_image.resize((600, h_size))

    image_bytes = io.BytesIO()
    resized.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    ocr_text = run_ocr_google(image_bytes)
    ocr_text = re.sub(r'(?<=[A-Z0-9])K(?=[A-Z0-9<])', '<', ocr_text)
    result = parse_mrz(ocr_text)

    return resized, ocr_text, result

uploaded_file = st.camera_input("Take a photo of your card")

if uploaded_file is not None:
    with st.spinner("Processing image..."):
        processed_img, ocr_text, parsed_data = process_image(uploaded_file)

    st.image(processed_img, caption="Processed Image", use_column_width=True)
    st.subheader("Extracted MRZ Text:")
    st.text_area("OCR Text", ocr_text, height=200)
    st.subheader("Parsed Data:")
    st.json(parsed_data)
