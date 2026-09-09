import streamlit as st
import requests
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

# --- PAGE SETUP ---
st.set_page_config(
    page_title="StockSense Pro | Enterprise Retail Vision",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN UI STYLING ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #2e3846;
    }
    .stMetric {
        background: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL CONSTANTS ---
CLASSES = ["Mineral Water", "Coca-Cola", "Pepsi", "Sprite", "Pure Milk"]
UNIT_PRICES = {
    "Mineral Water": 1.25,
    "Coca-Cola": 1.99,
    "Pepsi": 1.89,
    "Sprite": 1.79,
    "Pure Milk": 2.49
}
PAR_LEVELS = {
    "Mineral Water": 8,
    "Coca-Cola": 8,
    "Pepsi": 8,
    "Sprite": 6,
    "Pure Milk": 5
}

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=64)
st.sidebar.title("StockSense Pro")
st.sidebar.caption("Automated Shelf Auditing System v2.4")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Credentials")
ROBOFLOW_API_KEY = st.sidebar.text_input("Roboflow API Key", type="password", help="Enter your Roboflow private API key")
MODEL_ENDPOINT = st.sidebar.text_input("Model ID / Endpoint", value="stocksense-pro/1")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Filters")
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Cutoff (%)", min_value=10, max_value=100, value=40)
OVERLAP_THRESHOLD = st.sidebar.slider("Overlap / NMS (%)", min_value=10, max_value=100, value=30)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Adjust confidence down if dimly-lit shelf edges miss product detections.")

# --- MAIN DASHBOARD HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("📦 Store Inventory & Shelf Audit Portal")
    st.markdown("Real-time automated edge inventory tracking, out-of-stock anomaly alerts, and restock logistics.")
with col_head2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Store ID:** #SF-4089 | **Zone:** Beverages & Dairy")

st.markdown("---")

# --- IMAGE INGESTION ---
uploaded_file = st.file_uploader("Upload Retail Shelf Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    source_img = Image.open(uploaded_file).convert("RGB")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("📷 Ingested Scan")
        st.image(source_img, use_container_width=True)

    if st.button("🚀 Run Comprehensive Shelf Audit", type="primary"):
        if not ROBOFLOW_API_KEY:
            st.error("Please enter your Roboflow API key in the left sidebar to connect to your trained model.")
        else:
            with st.spinner("Processing computer vision inference and inventory metrics..."):
                buffered = io.BytesIO()
                source_img.save(buffered, format="JPEG")
                img_payload = buffered.getvalue()

                api_url = (
                    f"https://detect.roboflow.com/{MODEL_ENDPOINT}"
                    f"?api_key={ROBOFLOW_API_KEY}"
                    f"&confidence={CONFIDENCE_THRESHOLD}"
                    f"&overlap={OVERLAP_THRESHOLD}"
                )
                
                try:
                    res = requests.post(api_url, files={"file": img_payload})
                except Exception as e:
                    st.error(f"Network error connecting to inference endpoint: {e}")
                    st.stop()

                if res.status_code == 200:
                    payload = res.json()
                    predictions = payload.get("predictions", [])
                    
                    # Annotate bounding boxes
                    annotated_canvas = source_img.copy()
                    draw = ImageDraw.Draw(annotated_canvas)

                    detected_counts = {c: 0 for c in CLASSES}
                    
                    for p in predictions:
                        cls_name = p.get("class")
                        conf = p.get("confidence", 0.0)
                        x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
                        
                        x0 = x - (w / 2)
                        y0 = y - (h / 2)
                        x1 = x + (w / 2)
                        y1 = y + (h / 2)

                        draw.rectangle([x0, y0, x1, y1], outline="#00E676", width=4)
                        tag = f"{cls_name} ({int(conf * 100)}%)"
                        draw.rectangle([x0, max(0, y0 - 22), x0 + len(tag) * 8.5, y0], fill="#00E676")
                        draw.text((x0 + 4, max(0, y0 - 20)), tag, fill="#000000")

                        if cls_name in detected_counts:
                            detected_counts[cls_name] += 1
                        else:
                            detected_counts[cls_name] = 1

                    with col_img2:
                        st.subheader("🎯 Model Detections")
                        st.image(annotated_canvas, use_container_width=True)

                    # --- COMPUTE ANALYTICS ---
                    total_detected_units = sum(detected_counts.values())
                    total_inventory_val = sum(detected_counts[item] * UNIT_PRICES.get(item, 1.50) for item in detected_counts)
                    low_stock_items = [k for k, v in detected_counts.items() if 0 < v <= 3]
                    out_of_stock_items = [k for k, v in detected_counts.items() if v == 0]
                    
                    health_pct = int(((len(CLASSES) - len(out_of_stock_items) - (0.5 * len(low_stock_items))) / len(CLASSES)) * 100)

                    # --- EXECUTIVE METRICS ---
                    st.markdown("### 📈 Executive Performance KPIs")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Shelf Units", f"{total_detected_units} items", delta="Detected")
                    m2.metric("Audit Shelf Health", f"{health_pct}%", delta="-Action Needed" if health_pct < 75 else "Optimal")
                    m3.metric("Shelf Stock Value", f"${total_inventory_val:.2f}")
                    m4.metric("Attention Required", f"{len(out_of_stock_items) + len(low_stock_items)} SKUs", delta_color="inverse")

                    st.markdown("---")

                    # --- VISUALIZATION SECTION ---
                    st.markdown("### 📊 Shelf Capacity & Inventory Analytics")
                    chart_col1, chart_col2 = st.columns(2)

                    # Bar comparison chart: On-Shelf vs Recommended Par Level
                    chart_data = []
                    for product in CLASSES:
                        current_qty = detected_counts.get(product, 0)
                        par_qty = PAR_LEVELS.get(product, 8)
                        chart_data.append({"SKU": product, "Quantity": current_qty, "Type": "Current Stock"})
                        chart_data.append({"SKU": product, "Quantity": par_qty, "Type": "Target Par Level"})

                    df_chart = pd.DataFrame(chart_data)
                    fig_bar = px.bar(
                        df_chart, 
                        x="SKU", 
                        y="Quantity", 
                        color="Type", 
                        barmode="group",
                        title="Current Stock vs Target Par Levels",
                        color_discrete_map={"Current Stock": "#00C853", "Target Par Level": "#616161"}
                    )
                    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    chart_col1.plotly_chart(fig_bar, use_container_width=True)

                    # Donut chart: Stock share
                    fig_donut = px.pie(
                        names=list(detected_counts.keys()), 
                        values=list(detected_counts.values()),
                        title="SKU Distribution Share",
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_donut.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    chart_col2.plotly_chart(fig_donut, use_container_width=True)

                    # --- AUDIT SUMMARY TABLE ---
                    st.markdown("### 📋 SKU Audit & Automated Restock Ledger")
                    table_rows = []
                    for item in CLASSES:
                        count = detected_counts.get(item, 0)
                        par = PAR_LEVELS.get(item, 8)
                        reorder_qty = max(0, par - count)
                        cost = reorder_qty * UNIT_PRICES.get(item, 1.50)
                        
                        if count == 0:
                            status = "🔴 Out of Stock"
                        elif count <= 3:
                            status = "🟡 Low Stock"
                        else:
                            status = "🟢 Satisfactory"

                        table_rows.append({
                            "Product SKU": item,
                            "On-Shelf Count": count,
                            "Target Par": par,
                            "Status": status,
                            "Reorder Suggestion": f"{reorder_qty} units",
                            "Est. Restock Cost": f"${cost:.2f}"
                        })

                    df_table = pd.DataFrame(table_rows)
                    st.dataframe(df_table, use_container_width=True)

                    # --- EXPORT REPORT ---
                    csv = df_table.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Purchase Order & Audit Report (CSV)",
                        data=csv,
                        file_name="stocksense_audit_manifest.csv",
                        mime="text/csv"
                    )

                else:
                    st.error(f"Inference error {res.status_code}: {res.text}")
