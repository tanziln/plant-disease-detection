import streamlit as st
import numpy as np
import cv2 as cv
import keras

# ----------------- Configuration -----------------
st.set_page_config(page_title="🌿 plant Disease Detector", layout="centered")
    
# ----------------- Page Header -----------------
st.markdown(
"<h1 style='text-align: center; color: green;'>🌿 Leaf Disease Detector</h1>",
unsafe_allow_html=True
)

# ----------------- Optional Model Info -----------------
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    This model is built using deep learning and transfer learning techniques, trained on a dataset with 33 types of leaf diseases.

    ✅ **Supported Crops**:
    - Apple
    - Cherry
    - Corn
    - Grape
    - Peach
    - Pepper
    - Potato
    - Strawberry
    - Tomato

    Please upload a clear image of a single leaf for best results.
""")

# ----------------- Load Model & Labels -----------------
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

class_names = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']  # Replace with your actual class names
label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']
#st.write("""The leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model. The model is trained on a dataset containing images of 33 different types of leaf diseases. For more information about the architecture, dataset, and training process, please refer to the code and documentation provided.""")              

#st.write("Please input only leaf Images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato. Otherwise, the model will not work perfectly.")

#model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Fix color format
    img = cv.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    input_image = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(input_image)
    top_3 = np.argsort(predictions[0])[::-1][:3]  # Indices of top 3 predictions
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100
    predicted_label = class_names[predicted_index]

    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)    

    # Confidence filter and result display
    st.markdown("---")
    st.subheader("🧪 Final Diagnosis")

    if confidence < 80:
        st.warning("⚠️ Model is unsure about the result. Please try uploading a clear image of a single leaf.")
    else:
        if "healthy" in predicted_label.lower():
            st.success(f"🌿 Leaf is Healthy ({confidence:.2f}%)")
        else:
            st.error(f"🦠 Disease Detected: **{predicted_label}** ({confidence:.2f}%)")
        
