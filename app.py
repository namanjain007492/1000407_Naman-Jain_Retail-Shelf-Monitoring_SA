import streamlit as st
import requests
import io
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# PAGE SETUP & BRANDING
# ==========================================
st.set_page_config(
    page_title="StockSense Pro | Enterprise Retail AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Dashboard CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Card Containers */
    .saas-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .status-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-ok { background: rgba(35, 196, 131, 0.15); color: #2ecc71; border: 1px solid #2ecc71; }
    .badge-warn { background: rgba(241, 196, 15, 0.15); color: #f1c40f; border: 1px solid #f1c40f; }
    .badge-crit { background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid #e74c3c; }
    
    /* Polished Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d1117;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONFIGURATION & CATALOG DATA
# ==========================================
CATALOG = {
    "Mineral Water": {"category": "Hydration", "cost": 0.60, "retail": 1.29, "target_par": 10, "color": "#00d2d3"},
    "Coca-Cola":    {"category": "Carbonated", "cost": 0.85, "retail": 1.99, "target_par": 8,  "color": "#ff4757"},
    "Pepsi":        {"category": "Carbonated", "cost": 0.80, "retail": 1.89, "target_par": 8,  "color": "#2ed573"},
    "Sprite":       {"category": "Carbonated", "cost": 0.75, "retail": 1.79, "target_par": 6,  "color": "#ffa502"},
    "Pure Milk":    {"category": "Dairy",      "cost": 1.20, "retail": 2.49, "target_par": 6,  "color": "#70a1ff"}
}
CLASSES = list(CATALOG.keys())

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("## ⚡ StockSense Pro")
    st.caption("AI Retail Edge Vision System — v3.1")
    st.markdown("---")
    
    st.subheader("🔑 Inference Credentials")
    api_key = st.text_input("Roboflow API Key", type="password", help="Enter your private Roboflow API key")
    model_endpoint = st.text_input("Endpoint / Version", value="stocksense-pro/2")
    
    st.markdown("---")
    st.subheader("🎯 Vision Hyperparameters")
    confidence_threshold = st.slider("Detection Confidence (%)", 15, 95, 40)
    overlap_threshold = st.slider("Non-Max Suppression (%)", 10, 90, 30)
    
    st.markdown("---")
    st.caption("Deployment Node: **Active | Cloud Server**")
    st.caption("Camera Feed ID: **CAM-SHELF-04B**")

# ==========================================
# TOP HERO HEADER
# ==========================================
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.title("🛒 Autonomous Shelf Audit & Inventory Control")
    st.markdown("Live computer vision inference, planogram share analytics, and automated restocking dispatch.")
with h_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: right; color: #8b949e; font-size: 0.85rem;">
            <b>Store Branch:</b> North Zone Supercenter<br>
            <b>Aisle:</b> 04 (Beverages & Dairy)
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# INPUT PANEL (FILE UPLOADER)
# ==========================================
uploaded_file = st.file_uploader("📥 Ingest Shelf Scanner Capture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    source_img = Image.open(uploaded_file).convert("RGB")
    img_width, img_height = source_img.size

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### 📸 Raw Scan Preview")
        st.image(source_img, use_container_width=True)
        st.caption(f"Input Dimensions: {img_width} x {img_height} px")
    
    with c2:
        st.markdown("#### ⚙️ Analysis Actions")
        st.write("Run deep learning detection to inspect shelf facing, identify gaps, and calculate procurement deficits.")
        audit_trigger = st.button("🚀 Run Comprehensive AI Audit", type="primary", use_container_width=True)

    if audit_trigger:
        if not api_key:
            st.error("Missing Roboflow API Key. Enter your credential in the sidebar.")
            st.stop()

        start_time = time.time()
        
        # Prepare bytes
        buffered = io.BytesIO()
        source_img.save(buffered, format="JPEG", quality=95)
        img_bytes = buffered.getvalue()

        # Roboflow API Call
        api_url = (
            f"https://detect.roboflow.com/{model_endpoint}"
            f"?api_key={api_key}"
            f"&confidence={confidence_threshold}"
            f"&overlap={overlap_threshold}"
        )

        with st.spinner("🤖 Running vision inference and computing inventory health..."):
            try:
                response = requests.post(api_url, files={"file": img_bytes})
                latency = round((time.time() - start_time) * 1000, 1)
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"Roboflow API returned status {response.status_code}: {response.text}")
            st.stop()

        payload = response.json()
        predictions = payload.get("predictions", [])
        
        # Bounding Box Annotation with Case-Insensitive Catalog Normalization
        annotated_img = source_img.copy()
        draw = ImageDraw.Draw(annotated_img)
        detected_counts = {c: 0 for c in CLASSES}
        pred_records = []

        catalog_lookup = {k.lower(): k for k in CATALOG.keys()}

        for p in predictions:
            raw_c_name = p.get("class", "")
            conf = p.get("confidence", 0.0)
            x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
            
            normalized_key = catalog_lookup.get(raw_c_name.lower(), raw_c_name)
            c_name = normalized_key

            x0 = x - (w / 2)
            y0 = y - (h / 2)
            x1 = x + (w / 2)
            y1 = y + (h / 2)

            box_color = CATALOG.get(c_name, {}).get("color", "#2ecc71")
            draw.rectangle([x0, y0, x1, y1], outline=box_color, width=4)
            
            label_text = f"{c_name} {int(conf * 100)}%"
            draw.rectangle([x0, max(0, y0 - 22), x0 + len(label_text) * 8.5, y0], fill=box_color)
            draw.text((x0 + 4, max(0, y0 - 18)), label_text, fill="#000000")

            if c_name in detected_counts:
                detected_counts[c_name] += 1
            else:
                detected_counts[c_name] = 1

            pred_records.append({
                "SKU": c_name,
                "Confidence": conf,
                "X_Center": x,
                "Y_Center": y
            })

        # ==========================================
        # EXECUTIVE DASHBOARD TABS
        # ==========================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔎 Live Vision Audit", 
            "📊 Planogram & Category Insights", 
            "📋 Automated Procurement (PO)", 
            "⚡ Diagnostics & Telemetry"
        ])

        # ------------------------------------------
        # TAB 1: LIVE VISION AUDIT
        # ------------------------------------------
        with tab1:
            st.markdown("### 🖼️ Detection Results")
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.image(annotated_img, caption="Bounding Boxes & Confidence Labels", use_container_width=True)
            with res_col2:
                total_units = sum(detected_counts.values())
                stock_value = sum(detected_counts[k] * CATALOG[k]["retail"] for k in CLASSES)
                oos_count = sum(1 for k in CLASSES if detected_counts[k] == 0)
                low_count = sum(1 for k in CLASSES if 0 < detected_counts[k] <= 3)

                m1, m2 = st.columns(2)
                m1.metric("Total Items Detected", f"{total_units} units")
                m2.metric("Inventory Retail Value", f"${stock_value:.2f}")

                m3, m4 = st.columns(2)
                m3.metric("Critical Stockouts", f"{oos_count} SKUs", delta_color="inverse")
                m4.metric("Low Stock Alerts", f"{low_count} SKUs", delta_color="inverse")

                st.markdown("---")
                st.markdown("#### Real-time Stock Gauges")
                for item in CLASSES:
                    curr = detected_counts[item]
                    par = CATALOG[item]["target_par"]
                    fill_ratio = min(1.0, curr / par)
                    
                    st.write(f"**{item}** ({curr}/{par} units)")
                    st.progress(fill_ratio)

        # ------------------------------------------
        # TAB 2: PLANOGRAM & CATEGORY INSIGHTS
        # ------------------------------------------
        with tab2:
            st.markdown("### 📈 Visual Merchandise & Distribution")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                df_bar = []
                for k in CLASSES:
                    df_bar.append({"Product": k, "Quantity": detected_counts[k], "Metric": "Current On-Shelf"})
                    df_bar.append({"Product": k, "Quantity": CATALOG[k]["target_par"], "Metric": "Target Capacity"})
                df_bar_plot = pd.DataFrame(df_bar)

                fig_bars = px.bar(
                    df_bar_plot,
                    x="Product",
                    y="Quantity",
                    color="Metric",
                    barmode="group",
                    title="Actual Stock vs. Planogram Par Levels",
                    color_discrete_map={"Current On-Shelf": "#2ecc71", "Target Capacity": "#7f8c8d"}
                )
                fig_bars.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bars, use_container_width=True)

            with chart_col2:
                fig_donut = px.pie(
                    names=list(detected_counts.keys()),
                    values=list(detected_counts.values()),
                    title="Planogram Facing Share (%)",
                    hole=0.45,
                    color_discrete_sequence=[CATALOG[k]["color"] for k in CLASSES]
                )
                fig_donut.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_donut, use_container_width=True)

        # ------------------------------------------
        # TAB 3: PROCUREMENT & PURCHASE ORDERS
        # ------------------------------------------
        with tab3:
            st.markdown("### 📝 Smart Restock Ledger & Purchase Order Generation")
            
            po_data = []
            total_reorder_cost = 0.0
            
            for item in CLASSES:
                current = detected_counts[item]
                target = CATALOG[item]["target_par"]
                needed = max(0, target - current)
                cost = needed * CATALOG[item]["cost"]
                total_reorder_cost += cost

                if current == 0:
                    status = "🔴 OUT OF STOCK"
                elif current <= 3:
                    status = "🟡 LOW STOCK"
                else:
                    status = "🟢 SUFFICIENT"

                po_data.append({
                    "SKU": item,
                    "Category": CATALOG[item]["category"],
                    "On-Shelf": current,
                    "Target Par": target,
                    "Status": status,
                    "Restock Quantity": needed,
                    "Unit Cost": f"${CATALOG[item]['cost']:.2f}",
                    "Total Reorder Cost": f"${cost:.2f}"
                })

            df_po = pd.DataFrame(po_data)
            st.dataframe(df_po, use_container_width=True)

            c_po1, c_po2 = st.columns([2, 1])
            with c_po1:
                st.markdown(f"#### 💰 Total Procurement Cost: **${total_reorder_cost:.2f}**")
            with c_po2:
                csv_file = df_po.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Official PO (CSV)",
                    data=csv_file,
                    file_name="PO_Aisle4_InventoryManifest.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # ------------------------------------------
        # TAB 4: DIAGNOSTICS & TELEMETRY
        # ------------------------------------------
        with tab4:
            st.markdown("### ⚡ System Performance & Inspection Logs")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Inference Round-Trip", f"{latency} ms")
            col_t2.metric("Total Detections", len(predictions))
            col_t3.metric("Model Architecture", "Roboflow Fast (YOLO)")

            st.markdown("---")
            if pred_records:
                df_preds = pd.DataFrame(pred_records)
                st.markdown("#### Detections Confidence Distribution")
                fig_hist = px.histogram(
                    df_preds, 
                    x="Confidence", 
                    nbins=10, 
                    color="SKU",
                    title="Model Confidence Spread",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
