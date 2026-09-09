import streamlit as st
import requests
import io
from PIL import Image, ImageDraw

st.set_page_config(page_title="StockSense Pro", page_icon="🛒", layout="wide")

# ----------------- PAGE HEADER -----------------
st.title("🛒 StockSense Pro — Automated Inventory Monitoring")
st.markdown("Upload a retail shelf image to detect products, count stock, and generate restocking alerts.")

# ----------------- CONFIGURATION SIDEBAR -----------------
st.sidebar.header("⚙️ Model Configuration")
ROBOFLOW_API_KEY = st.sidebar.text_input("Roboflow API Key", type="password")
MODEL_ENDPOINT = st.sidebar.text_input("Model ID / Endpoint", value="stocksense-pro/1")
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", min_value=10, max_value=100, value=40)

CLASSES = ["Mineral Water", "Coca-Cola", "Pepsi", "Sprite", "Pure Milk"]

# ----------------- HELPER: STOCK STATUS -----------------
def get_stock_status(count):
    if count == 0:
        return "Out of Stock", "error"
    elif 1 <= count <= 3:
        return "Low Stock", "warning"
    else:
        return "In Stock", "success"

# ----------------- FILE UPLOADER -----------------
uploaded_file = st.file_uploader("Choose a shelf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("Run Inventory Analysis", type="primary"):
        if not ROBOFLOW_API_KEY:
            st.error("Please provide your Roboflow API Key in the sidebar.")
        else:
            with st.spinner("Analyzing shelf stock..."):
                # Prepare image for API
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()

                # Call Roboflow Inference API
                upload_url = f"https://detect.roboflow.com/{MODEL_ENDPOINT}?api_key={ROBOFLOW_API_KEY}&confidence={CONFIDENCE_THRESHOLD}"
                response = requests.post(upload_url, files={"file": img_bytes})

                if response.status_code == 200:
                    predictions = response.json().get("predictions", [])

                    # Draw bounding boxes
                    annotated_image = image.copy()
                    draw = ImageDraw.Draw(annotated_image)
                    
                    counts = {c: 0 for c in CLASSES}

                    for pred in predictions:
                        label = pred.get("class")
                        confidence = pred.get("confidence", 0)
                        x = pred.get("x")
                        y = pred.get("y")
                        w = pred.get("width")
                        h = pred.get("height")

                        x0 = x - w / 2
                        y0 = y - h / 2
                        x1 = x + w / 2
                        y1 = y + h / 2

                        draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=3)
                        draw.text((x0, max(0, y0 - 15)), f"{label} {confidence:.2f}", fill="#00FF00")

                        if label in counts:
                            counts[label] += 1
                        else:
                            counts[label] = counts.get(label, 0) + 1

                    with col2:
                        st.subheader("Detections")
                        st.image(annotated_image, use_container_width=True)

                    st.markdown("---")
                    st.header("📊 Inventory Breakdown & Restock Alerts")

                    # Display metric cards
                    metric_cols = st.columns(len(CLASSES))
                    for idx, prod in enumerate(CLASSES):
                        qty = counts.get(prod, 0)
                        status_text, status_type = get_stock_status(qty)

                        with metric_cols[idx]:
                            st.metric(label=prod, value=f"{qty} units")
                            if status_type == "error":
                                st.error(status_text)
                            elif status_type == "warning":
                                st.warning(status_text)
                            else:
                                st.success(status_text)

                    # Summary alerts
                    st.markdown("### 🔔 Operational Action Plan")
                    restock_needed = [p for p, q in counts.items() if q <= 3]
                    if restock_needed:
                        st.warning(f"Restock Required for: {', '.join(restock_needed)}")
                    else:
                        st.success("All monitored inventory levels are sufficient.")

                else:
                    st.error(f"Inference failed with status code {response.status_code}: {response.text}")
