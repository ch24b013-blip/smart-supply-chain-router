import streamlit as st
import torch
import os
from torchvision import transforms
from PIL import Image
from model import BananaNet
from supply_chain import get_network_graph, compute_shortest_paths

# --- UI Setup (Blinkit Style) ---
st.set_page_config(page_title="Quick-Comm AI Logistics", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight: bold;}
    .success-text {color: #00b894;}
    .danger-text {color: #d63031;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ AI Quick-Commerce Logistics Hub")
st.markdown("Automated Perishable Routing & Distribution Network")

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = BananaNet()
    if os.path.exists("banana_net_v2.pth"):
        model.load_state_dict(torch.load("banana_net_v2.pth", map_location=device, weights_only=True))
    model.eval()
    return model

model = load_model()
vision_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Main Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<p class='big-font'>📸 Intake Scanner</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop payload image here...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

# --- Logistics Execution ---
if uploaded_file is not None:
    # 1. AI Prediction
    # 1. AI Prediction (Intercepting the backwards logic)
    img_tensor = vision_pipeline(img).unsqueeze(0)
    with torch.no_grad():
        predicted_age = model(img_tensor).item()
        
        # SDE Fix: Invert the 'Age' into a 'Countdown to Death'
        # Assuming a standard banana goes completely black around Day 10.
        # We use max() to ensure it never predicts a negative shelf life.
        predicted_shelf_life = max(0.0, 10.0 - predicted_age)
    
    with col2:
        st.markdown("<p class='big-font'>🌍 Live Network Map</p>", unsafe_allow_html=True)
        st.metric(label="Predicted Shelf Life", value=f"{predicted_shelf_life:.2f} Days")
        
        # 2. DSA Routing
        graph = get_network_graph()
        transit_times, optimal_paths = compute_shortest_paths(graph, 'Central Warehouse')
        
        # 3. Dynamic Visual Map Generation (Mermaid.js)
        mermaid_code = "graph LR\n"
        
        # Draw edges with transit times
        for node, neighbors in graph.items():
            for neighbor, time in neighbors:
                # Remove spaces for Mermaid node IDs
                n1, n2 = node.replace(" ", "_"), neighbor.replace(" ", "_")
                mermaid_code += f"    {n1}[\"{node}\"] -- {time} days --> {n2}[\"{neighbor}\"]\n"
        
        # Color code the nodes based on AI prediction!
        for node, time_taken in transit_times.items():
            n_id = node.replace(" ", "_")
            if time_taken == 0:
                mermaid_code += f"    style {n_id} fill:#0984e3,stroke:#fff,stroke-width:2px,color:#fff\n" # Blue Warehouse
            elif predicted_shelf_life > time_taken:
                mermaid_code += f"    style {n_id} fill:#00b894,stroke:#fff,stroke-width:2px,color:#fff\n" # Green (Safe)
            else:
                mermaid_code += f"    style {n_id} fill:#d63031,stroke:#fff,stroke-width:2px,color:#fff\n" # Red (Spoiled)

        # Render the map
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

        # 4. Routing Action Plan
        st.markdown("### 🚚 Automated Action Plan")
        
        # Filter endpoints (stores with no outgoing connections)
        endpoints = [node for node, edges in graph.items() if len(edges) == 0]
        
        cols = st.columns(3) # Grid layout for Blinkit feel
        col_idx = 0
        
        for node in endpoints:
            time_taken = transit_times[node]
            remaining = predicted_shelf_life - time_taken
            path = " ➔ ".join(optimal_paths[node])
            
            with cols[col_idx % 3]:
                if remaining > 0:
                    st.success(f"**{node}**\n\n✅ Safe\n\nTransit: {time_taken:.1f}d\n\nMargin: +{remaining:.2f}d")
                else:
                    st.error(f"**{node}**\n\n❌ Reject\n\nTransit: {time_taken:.1f}d\n\nSpoils: {-remaining:.2f}d early")
            col_idx += 1
else:
    with col2:
        st.info("System Standby. Awaiting computer vision intake to render dynamic routing map.")