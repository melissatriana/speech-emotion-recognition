import os
import tempfile

import streamlit as st

from utils import predict_emotion

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# ==================================================
# HEADER
# ==================================================

st.title("🎤 Speech Emotion Recognition")

st.caption(
    "Detect emotions from your voice using a fine-tuned Wav2Vec2 model."
)

st.divider()

# ==================================================
# UPLOAD
# ==================================================

st.subheader("Upload Audio (.wav)")

uploaded_file = st.file_uploader(
    "Upload your audio (.wav)",
    type=["wav"]
)

# ==================================================
# MAIN
# ==================================================

if uploaded_file is not None:

    st.subheader("🎧 Audio Preview")
    st.audio(uploaded_file)

    st.write("")

    if st.button(
        "🎯 Predict Emotion",
        use_container_width=True
    ):

        with st.spinner("Analyzing audio..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            emotion, confidence, probs = predict_emotion(
                temp_path
            )

            os.remove(temp_path)

        st.divider()

        # ==========================================
        # RESULT
        # ==========================================

        st.success("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Emotion",
                emotion
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.divider()

        # ==========================================
        # PROBABILITY
        # ==========================================

        st.subheader("📊 Emotion Probability")

        sorted_probs = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for emo, value in sorted_probs:

            col1, col2 = st.columns([4,1])

            with col1:
                st.write(f"**{emo}**")

            with col2:
                st.write(f"**{value*100:.2f}%**")

            st.progress(float(value))

        st.divider()
