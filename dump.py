import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os
import json
from datetime import datetime

# 1. SETUP PAGE CONFIGURATION
st.set_page_config(page_title="AI text to Image Generator", layout="wide")

# 2. CREATE OUTPUT FOLDER IF NOT EXISTS
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

# 3. LOAD THE MODEL (This runs once and caches the model)
@st.cache_resource

def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    return pipe, device
# Show loading spinner while model downloads (takes time on first run)
with st.spinner("Loading Model... (First run requires downloading ~4GB)"):
    pipe, device = load_model()

# 4. SIDEBAR - SETTINGS
st.sidebar.header("Settings")
st.sidebar.write(f"Running on: **{device.upper()}**")
num_inference_steps = st.sidebar.slider("Steps (Quality)", 10, 40, 20)
guidance_scale = st.sidebar.slider("Guidance Scale (Creativity)", 1.0, 20.0, 7.5)
num_images = st.sidebar.slider("Number of Images", 1, 2, 1)

# 5. MAIN INTERFACE
st.title(" AI text to Image Generator")
st.markdown("Enter a description below to generate an image.")

prompt = st.text_area("Prompt", placeholder="A cyberpunk city with neon lights, 8k resolution...")
negative_prompt = st.text_input("Negative Prompt (What to avoid)", placeholder="blurry, distorted, ugly")

if st.button("Generate Image", type="primary"):
    if not prompt:
        st.error("Please enter a prompt!")
    else:
        with st.spinner(f"Generating... (Running on {device.upper()})"):
            # Generate the image
            images = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            height=512,
            width=512
            ).images

            # Display and Save
            cols = st.columns(num_images)
            for idx, image in enumerate(images):
                # Save Image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{idx}.png"
                filepath = os.path.join(output_folder, filename)
                image.save(filepath)

                # Save Metadata (Requirement)
                meta_data = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "steps": num_inference_steps,
                    "guidance": guidance_scale,
                    "timestamp": timestamp
                }
                with open(filepath.replace(".png", ".json"), "w") as f:
                    json.dump(meta_data, f)

                # Show in UI
                with cols[idx]:
                    st.image(image, caption="Generated Image")
                    with open(filepath, "rb") as file:
                        st.download_button(
                            label="Download",
                            data=file,
                            file_name=filename,
                            mime="image/png"
                        )
            st.success(f"Saved to /{output_folder}")