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
# EMOJI
# ==================================================

EMOJI = {
    "Neutral": "😐",
    "Happy": "😊",
    "Surprise": "😲",
    "Disgust": "🤢",
    "Disappointed": "😞"
}

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
    "",
    type=["wav"],
    label_visibility="collapsed"
)

st.info(
    "📁 Upload a **.wav** audio file to begin emotion prediction."
)

# ==================================================
# MAIN
# ==================================================

if uploaded_file is not None:

    with st.expander("🎧 Audio Preview", expanded=True):
        st.audio(uploaded_file)

    st.write("")

    if st.button(
        "🚀 Analyze Emotion",
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

        st.write("")

        # ==========================================
        # RESULT
        # ==========================================

        st.success(
            f"Prediction completed successfully! Detected emotion: **{EMOJI[emotion]} {emotion}**"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Emotion",
                f"{EMOJI[emotion]} {emotion}"
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

            st.write(f"**{EMOJI[emo]} {emo}**")

            st.progress(float(value))

            st.caption(
                f"{value*100:.2f}%"
            )

        st.divider()
