

import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import torch
import base64

# Load Hugging Face Models with optimized settings
@st.cache_resource
def load_models():
    # Load the text-to-text generation model (Hugging Face)
    llm_model = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)
    
    # Load Stable Diffusion for text-to-image generation
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision="fp16" if torch.cuda.is_available() else None,
    )
    if torch.cuda.is_available():
        image_gen_model = image_gen_model.to("cuda")
    else:
        image_gen_model = image_gen_model.to("cpu")  # Fallback to CPU if no GPU is available
    return llm_model, image_gen_model

# Function to encode background image in base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to apply background image and text color
def add_custom_styles():
    image_path = "llm_banner.jpg"  # File is in the root directory
    base64_image = get_base64_image(image_path)
    custom_style = f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{base64_image});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    h1, h3, label {{
        color: white !important;
    }}
    .stTextInput > div > label {{
        color: white !important;
    }}
    input[type="text"] {{
        color: black !important;
        background-color: transparent !important;
        border: 1px solid white !important;
    }}
    input::placeholder {{
        color: white !important;
        opacity: 1 !important;
    }}
    .stSpinner > div > div {{
        color: yellow !important;
    }}
    </style>
    """
    st.markdown(custom_style, unsafe_allow_html=True)

# Initialize models
llm_model, image_model = load_models()

# Function to refine prompt using Hugging Face model
def refine_prompt_hf(query):
    prompt = f"Refine this query for image generation: {query}"
    result = llm_model(prompt, max_length=50)
    refined_prompt = result[0]["generated_text"]
    return refined_prompt.strip()

# Function to generate an image
def generate_image(prompt):
    with st.spinner("Generating image... Please wait!"):
        image = image_model(prompt).images[0]
    return image

# Add custom background and text styles
add_custom_styles()

# Streamlit App UI
st.title("LLM Powered Text to Image Generator")

# User Input
user_query = st.text_input("Describe your image")

# Generate Button
if st.button("Generate Image"):
    # Refine the input query
    refined_query = refine_prompt_hf(user_query)

    # Generate the image
    generated_image = generate_image(refined_query)

    # Display Image using Matplotlib
    st.write("### Generated Image:")
    fig, ax = plt.subplots()
    ax.imshow(generated_image)
    ax.axis("off")
    st.pyplot(fig)

    # Save and Add a download button
    generated_image.save("generated_image.png")
    with open("generated_image.png", "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="generated_image.png",
            mime="image/png"
        )
