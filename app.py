import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import random

# App title and description
st.title("Stable Diffusion Image Generator")
st.markdown("Generate images using Stable Diffusion model via HuggingFace Space API")

# Sidebar for parameters
st.sidebar.header("Generation Parameters")

# Text input parameters
prompt = st.sidebar.text_area("Prompt", "a photograph of an astronaut riding a horse", help="Describe what you want to generate")
negative_prompt = st.sidebar.text_area("Negative Prompt", "blurry, bad quality", help="Describe what you want to avoid")

# Numerical parameters
num_inference_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=50, value=35, help="Higher values = more detailed but slower")
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=2.0, step=0.5, help="How closely to follow the prompt")

# Image dimensions
col1, col2 = st.sidebar.columns(2)
with col1:
    width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
with col2:
    height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)

# Seed for reproducibility
use_random_seed = st.sidebar.checkbox("Use random seed", value=True)
if use_random_seed:
    seed = random.randint(1, 999999)
    st.sidebar.text(f"Random Seed: {seed}")
else:
    seed = st.sidebar.number_input("Seed", min_value=1, max_value=999999, value=42)

# HuggingFace token input (for authenticated requests if needed)
use_auth = st.sidebar.checkbox("Use HuggingFace Token", value=False)
hf_token = None
if use_auth:
    hf_token = st.sidebar.text_input("HuggingFace Token", type="password")
    if not hf_token:
        st.sidebar.warning("Please enter your HuggingFace token")

# Generation button
generate_button = st.sidebar.button("Generate Image", type="primary")

# Function to generate image
def generate_image(params, token=None):
    api_url = "https://stpete2-stable-diffusion-FastApi.hf.space/generate"
    
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    with st.spinner("Generating image..."):
        try:
            response = requests.post(api_url, json=params, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result["image"])
                image = Image.open(io.BytesIO(image_data))
                return image, result["parameters"]
            else:
                st.error(f"Error: {response.status_code}")
                st.text(response.text)
                return None, None
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            return None, None

# Main content area with placeholder for the generated image
image_placeholder = st.empty()
params_placeholder = st.empty()

# If the button is clicked, generate the image
if generate_button:
    # Prepare parameters
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "height": height,
        "width": width,
        "seed": seed
    }
    
    # Generate image
    image, parameters = generate_image(payload, hf_token)
    
    if image:
        # Display image
        image_placeholder.image(image, caption="Generated Image", use_column_width=True)
        
        # Display parameters used
        params_placeholder.json(parameters)
        
        # Add download button
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )
else:
    # Display instructions when app first loads
    image_placeholder.info("Configure your parameters in the sidebar and click 'Generate Image' to create an image.")
