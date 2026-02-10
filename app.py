import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import replicate

# ================= LOAD ENV =================
load_dotenv()

# Groq client (text recommendations)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Replicate requires token as env variable
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

# ================= STREAMLIT CONFIG =================
st.set_page_config(
    page_title="StyleSense",
    layout="wide"
)

# ================= SESSION STATE =================
if "page" not in st.session_state:
    st.session_state.page = "input"

# ================= IMAGE STYLE ANALYSIS =================
def analyze_image_style(image_file):
    image = Image.open(image_file).resize((100, 100))
    pixels = np.array(image).astype(int)

    brightness = pixels.mean()
    if brightness < 85:
        light = "Dark"
    elif brightness < 170:
        light = "Balanced"
    else:
        light = "Light"

    r, g, b = pixels.mean(axis=(0, 1))
    if r > g + 20 and r > b + 20:
        color = "Red tones"
    elif b > r + 20 and b > g + 20:
        color = "Blue tones"
    elif g > r + 20 and g > b + 20:
        color = "Green tones"
    else:
        color = "Neutral tones"

    contrast_val = pixels.std()
    if contrast_val > 60:
        contrast = "High contrast"
    elif contrast_val > 30:
        contrast = "Medium contrast"
    else:
        contrast = "Low contrast"

    return light, color, contrast

# ================= IMAGE GENERATION (REPLICATE) =================
def generate_outfit_image(outfit_description):
    """
    Generates an outfit preview using Replicate.
    Returns image bytes or URL that Streamlit can render.
    """

    prompt = f"""
Fashion catalog photography of a model wearing the following outfit:

{outfit_description}

Clean background, studio lighting, realistic fabric texture,
high-quality fashion photography.
"""

    output = replicate.run(
        "google/imagen-4",
        input={
            "prompt": prompt
        }
    )

    # Case 1: Replicate returns a FileOutput object
    if hasattr(output, "read"):
        return output.read()

    # Case 2: Replicate returns a list
    if isinstance(output, list):
        return output[0]

    # Case 3: Replicate returns a URL string
    return output

# ================= PAGE 1: INPUT =================
if st.session_state.page == "input":

    st.title("StyleSense")
    st.write("Upload your outfit and personalize your recommendations")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader(
            "Upload an outfit image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:
            st.image(uploaded_image, width=350)

    with col2:
        preferred_style = st.selectbox(
            "Preferred style",
            ["No preference", "Casual", "Streetwear", "Formal", "Sporty", "Minimal", "Ethnic"]
        )

        climate = st.selectbox(
            "Climate",
            ["No preference", "Hot", "Cold", "Mild", "Rainy"]
        )

        occasion = st.selectbox(
            "Occasion",
            ["No preference", "Daily wear", "Office", "Party", "Travel", "Date", "Festival"]
        )

        fit = st.selectbox(
            "Fit preference",
            ["No preference", "Slim", "Regular", "Oversized"]
        )

        gender = st.selectbox(
            "Gender",
            ["No preference", "Men", "Women", "Unisex"]
        )

        generate = st.button("Generate Outfit")

    if generate and uploaded_image:
        st.session_state.uploaded_image = uploaded_image
        st.session_state.preferred_style = preferred_style
        st.session_state.climate = climate
        st.session_state.occasion = occasion
        st.session_state.fit = fit
        st.session_state.gender = gender
        st.session_state.page = "result"
        st.rerun()

# ================= PAGE 2: RESULTS =================
if st.session_state.page == "result":

    st.title("Your Personalized Outfit")

    col1, col2 = st.columns([1, 1.7])

    # ---------- LEFT: USER IMAGE ----------
    with col1:
        st.subheader("Uploaded Outfit")
        st.image(st.session_state.uploaded_image, width=350)

    # ---------- RIGHT: AI OUTPUT ----------
    with col2:
        light, color, contrast = analyze_image_style(
            st.session_state.uploaded_image
        )

        st.subheader("Detected Style")
        st.markdown(f"""
- **Color family:** {color}  
- **Brightness:** {light}  
- **Contrast:** {contrast}
""")

        st.subheader("Your Preferences")
        st.markdown(f"""
- **Style:** {st.session_state.preferred_style}  
- **Climate:** {st.session_state.climate}  
- **Occasion:** {st.session_state.occasion}  
- **Fit:** {st.session_state.fit}  
- **Gender:** {st.session_state.gender}
""")

        tabs = st.tabs(["Everyday Look", "Elevated Look", "Bold Look"])

        for index, tab in enumerate(tabs):
            with tab:
                variation = ["Everyday", "Elevated", "Bold"][index]

                prompt = f"""
You are a professional fashion stylist AI.

Detected outfit details:
- Color family: {color}
- Brightness: {light}
- Contrast: {contrast}

User preferences:
- Style: {st.session_state.preferred_style}
- Climate: {st.session_state.climate}
- Occasion: {st.session_state.occasion}
- Fit: {st.session_state.fit}
- Gender: {st.session_state.gender}

Variation type: {variation}

Respond EXACTLY in this format:

Topwear:
Bottomwear:
Footwear:
Accessory:
Why it works:
"""

                response = groq_client.chat.completions.create(
                    model="Llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

                outfit_text = response.choices[0].message.content
                st.write(outfit_text)

                if st.button(f"Generate Outfit Preview ({variation})"):
                    with st.spinner("Generating AI outfit preview..."):
                        image_output = generate_outfit_image(outfit_text)
                        st.subheader("AI-Generated Outfit Preview")
                        st.image(image_output)
                        st.caption(
                            "AI-generated outfit preview for visualization purposes. "
                            "This is not a virtual try-on."
                        )

        if st.button("⬅ Back to Upload"):
            st.session_state.page = "input"
            st.rerun()


