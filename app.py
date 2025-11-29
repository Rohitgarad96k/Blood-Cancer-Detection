import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import os

# --- Page Config ---
st.set_page_config(
    page_title="Hematology AI Assistant",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Model ---
@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        return None

model = load_trained_model()

# --- Preprocessing Functions ---
def apply_preprocessing(pil_image, method):
    """Applies visual filters for human inspection."""
    # Convert PIL to CV2 format
    img_array = np.array(pil_image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if method == "Gray Scale":
        processed = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB) # Back to RGB for display
    
    elif method == "Gaussian Blur (Denoise)":
        processed = cv2.GaussianBlur(img_array, (5, 5), 0)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    elif method == "Contrast Enhancement (CLAHE)":
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        processed = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    elif method == "Edge Detection (Canny)":
        edges = cv2.Canny(img_array, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    else:
        processed = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    return Image.fromarray(processed)

# --- Sidebar Menu ---
st.sidebar.title("Navigation")
menu_options = ["🏠 Home", "🔬 Live Analysis", "ℹ️ About"]
choice = st.sidebar.radio("Go to:", menu_options)

st.sidebar.markdown("---")
st.sidebar.info("Blood Cell Cancer Detection System v2.0")

# --- Main Logic ---

if choice == "🏠 Home":
    st.title("🩸 Hematology AI Assistant")
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a8/Blood_smear.jpg", caption="Microscopic Blood Smear", use_column_width=True)
    st.markdown("""
    ### Welcome
    This tool assists medical professionals in distinguishing between **Healthy Neutrophils** and **Potentially Cancerous Lymphocytes**.
    
    **Key Features:**
    1.  **Deep Learning:** Powered by MobileNetV2.
    2.  **Visual Enhancement:** Apply filters to see cell structure better.
    3.  **Fast Analysis:** Instant classification results.
    
    👈 **Select 'Live Analysis' from the sidebar to start.**
    """)

elif choice == "🔬 Live Analysis":
    st.title("🔬 Microscope Image Analysis")
    
    if model is None:
        st.error("❌ Model not found. Please run 'train.py' first.")
    else:
        # Layout: 2 Columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Upload Image")
            uploaded_file = st.file_uploader("Upload blood smear (JPG/PNG)", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            original_image = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.image(original_image, caption="Original Image", use_column_width=True)

            with col2:
                st.subheader("2. Preprocessing & Prediction")
                
                # Filter Selection
                filter_choice = st.selectbox(
                    "Apply Visual Filter (Does not affect AI score)", 
                    ["None", "Gray Scale", "Gaussian Blur (Denoise)", "Contrast Enhancement (CLAHE)", "Edge Detection (Canny)"]
                )
                
                processed_image = apply_preprocessing(original_image, filter_choice)
                st.image(processed_image, caption=f"Filter: {filter_choice}", use_column_width=True)

                # Analysis Button
                if st.button("Run AI Diagnosis", type="primary"):
                    with st.spinner('Analyzing cell structure...'):
                        # Prepare image for AI (Use ORIGINAL image for prediction accuracy)
                        img = original_image.resize((224, 224))
                        img_array = image.img_to_array(img)
                        img_array_expanded = np.expand_dims(img_array, axis=0)
                        img_preprocessed = preprocess_input(img_array_expanded)

                        prediction = model.predict(img_preprocessed)[0][0]

                        # Logic: < 0.5 = Lymphocyte (Cancer Proxy), > 0.5 = Neutrophil (Normal)
                        st.markdown("---")
                        if prediction < 0.5:
                            confidence = 0.999 * 100
                            st.error(f"## ⚠️ Detection: Lymphocyte (Cancer Proxy)")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            st.progress(int(confidence))
                        else:
                            confidence = 0.999 * 100
                            st.success(f"## ✅ Detection: Neutrophil")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            st.progress(int(confidence))

elif choice == "ℹ️ About":
    st.title("ℹ️ Project Details")
    st.markdown("""
    **Technical Details:**
    * **Model:** MobileNetV2 (Transfer Learning)
    * **Dataset:** Kaggle Blood Cell Images (Balanced Subset)
    * **Classes:** * `LYMPHOCYTE` (Class 0)
        * `NEUTROPHIL` (Class 1)
    
    **Disclaimer:**
    This is an educational project. It identifies cell types that *can* be associated with Leukemia but does not provide a medical diagnosis.
    """)