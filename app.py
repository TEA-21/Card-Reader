import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import re

# Setup
st.set_page_config(page_title="Card OCR Scanner", layout="centered")
st.title("📇 Card OCR Scanner (Google OCR Edition)")

# Load Google credentials from secrets
google_creds = json.loads(st.secrets["GOOGLE_CREDENTIALS"])

# Build API request headers
api_url = "https://vision.googleapis.com/v1/images:annotate"
api_key = google_creds.get("private_key_id")  # Just a dummy to force validation

# OCR function using Google Cloud Vision
def google_ocr(image_file):
    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    body = {
        "requests": [
            {
                "image": {"content": image_base64},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    response = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={google_creds['private_key_id']}",
        headers={"Content-Type": "application/json"},
        json=body
    )

    result = response.json()
    try:
        text = result["responses"][0]["fullTextAnnotation"]["text"]
        return text
    except Exception as e:
        st.error("Error parsing OCR response.")
        st.stop()

# Parse MRZ (same as before)
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

# Streamlit UI
uploaded_file = st.camera_input("Take a photo of your card") or st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Running OCR using Google Vision..."):
        ocr_text = google_ocr(uploaded_file)
        parsed_data = parse_mrz(ocr_text)

    st.subheader("Extracted MRZ Text:")
    st.text_area("OCR Result", ocr_text, height=200)

    st.subheader("Parsed Data:")
    st.json(parsed_data)