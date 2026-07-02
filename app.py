import streamlit as st
from PIL import Image

from utils import (
    load_model,
    get_val_transform,
    predict_probability,
)

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------

st.set_page_config(
    page_title="SalesCode AI",
    page_icon="🖥️",
    layout="wide"
)

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#2563EB;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:25px;
}

.metric-card{
    background:#F8FAFC;
    padding:20px;
    border-radius:12px;
    border:1px solid #E5E7EB;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def get_model():
    return load_model("best_model.pth")


model = get_model()
transform = get_val_transform()

# ----------------------------------------------------
# Header
# ----------------------------------------------------

st.markdown(
    '<div class="title">🖥️ SalesCode AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real Photo vs Screen Photo Classification</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("About")

st.sidebar.info(
    """
This application detects whether an uploaded image is:

📷 **Real Photograph**

or

💻 **Photo of a Screen**

Model:
MobileNetV3-Small

Framework:
PyTorch
"""
)

# ----------------------------------------------------
# Upload
# ----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    image_tensor = transform(image)

    probability = predict_probability(
        model,
        image_tensor
    )

    col1, col2 = st.columns([1.2,1])

    # ----------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # ----------------------------------------

    with col2:

        st.subheader("Prediction")

        if probability >= 0.5:

            st.error("💻 Photo of a Screen")

            confidence = probability

        else:

            st.success("📷 Real Photograph")

            confidence = 1 - probability

        st.markdown("### Screen Probability")

        st.progress(float(probability))

        st.metric(
            "Screen Probability",
            f"{probability*100:.2f}%"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence*100:.2f}%"
        )

        st.markdown("---")

        st.subheader("Interpretation")

        st.write("**0 → Real Photograph**")

        st.write("**1 → Photo of a Screen**")

        st.markdown("---")

        if probability >= 0.5:

            st.warning(
                "The model predicts that this image is likely a photo captured from a display screen."
            )

        else:

            st.success(
                "The model predicts that this image is likely a real-world photograph."
            )

# ----------------------------------------------------
# Footer
# ----------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div class="footer">

Built with ❤️ using

<b>PyTorch</b> • <b>MobileNetV3-Small</b> • <b>Streamlit</b>

</div>
""",
unsafe_allow_html=True
)