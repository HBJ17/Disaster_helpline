import streamlit as st
import tempfile
import os

# Import the main processing function from your backend script
from analysis_pipeline import process_audio_file

# --- Page Configuration ---
st.set_page_config(
    page_title="Emergency AI Analysis",
    page_icon="🚨",
    layout="wide"
)

# --- UI Layout ---
st.title("🚨 Emergency AI Analysis")
st.markdown("Upload an audio file (.wav, .mp3) to analyze its emotional content and distress level.")

# --- File Uploader and Options ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
fast_mode = st.sidebar.checkbox("Enable Fast Mode", help="Skips heavy, chunked audio analysis for a quicker result.")

# --- Main Logic ---
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.sidebar.button("Analyze Audio"):
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filepath = tmp_file.name

        try:
            # Show a spinner while processing
            with st.spinner('Analyzing audio... This may take a moment.'):
                results = process_audio_file(temp_filepath, fast_mode=fast_mode)

            # Display results once processing is complete
            st.subheader("📊 Analysis Report")

            if results.get("error"):
                st.error(f"An error occurred: {results['error']}")
            else:
                # Display key metrics in columns
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Distress Token", 
                    results['distress'].upper(),
                    help="The overall assessed distress level."
                )
                col2.metric(
                    "Final Emotion",
                    results['emotion'].upper(),
                    help="The dominant emotion detected in the call."
                )
                col3.metric(
                    "Confidence Score",
                    f"{results['confidence']:.2f}",
                    help="The model's confidence in the detected emotion."
                )

                # Display the transcript in an expander
                with st.expander("📜 View Full Transcript"):
                    st.write(results['transcript'] or '(No speech was detected)')

                # Display escalation reason if available
                if results.get('reason'):
                    st.info(f"💡 **Reason for Distress Level:** {results['reason']}")
                
                st.success("Analysis complete!")
                #st.balloons()

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
else:
    st.info("Please upload an audio file and click 'Analyze Audio' to begin.")