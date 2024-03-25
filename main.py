import streamlit as st

st.header("Welcome to the Soap Recognizer")
st.subheader("NOTE: Liquid Soap is not soap, it is detergent")

uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_img is not None:
    # Display the uploaded image
    st.image(uploaded_img, caption='Uploaded Image.', use_column_width=True)