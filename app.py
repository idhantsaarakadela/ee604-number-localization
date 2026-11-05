from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from inference_pipeline import NumericalTextDetector

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"


@st.cache_resource
def load_detector():
    return NumericalTextDetector(str(WEIGHTS), confidence=0.25, imgsz=640)


st.set_page_config(page_title="Locating numbers in text images", page_icon="🔢")
st.title("Locating numbers in text images")
st.write("Upload a JPG or PNG showing one house number, or try a sample from SVHN.")

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
samples = sorted((ROOT / "examples").glob("*.jpg"))
sample = st.selectbox(
    "Or choose a sample",
    [None, *samples],
    format_func=lambda path: path.name if path else "Select a sample",
    disabled=upload is not None,
)

if upload is not None:
    image = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
elif sample is not None:
    image = cv2.imread(str(sample))
else:
    st.stop()

if image is None:
    st.error("The image could not be read. Try another JPG or PNG.")
    st.stop()

with st.spinner("Reading the number…"):
    result = load_detector().predict(image)

sequence = next(iter(result["sequences"]), None)
if sequence:
    st.metric("Predicted number", sequence["text"])
    st.caption(f"Mean digit confidence: {sequence['confidence']:.1%}")
else:
    st.info("No number detected.")

left, right = st.columns(2)
left.image(image, channels="BGR", caption="Input", use_container_width=True)
# Draw the overlay without text covering the digits.
overlay = image.copy()
for digit in result["digits"]:
    x1, y1, x2, y2 = map(round, digit["bbox"])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), 1)
if sequence:
    x1, y1, x2, y2 = map(round, sequence["bbox"])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 1)
right.image(
    overlay, channels="BGR", caption="Blue: digits · Green: number",
    use_container_width=True,
)
