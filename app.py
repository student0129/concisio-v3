import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Add debugging
print("Starting Concisio App...")
print(f"Environment variables loaded: HF_TOKEN={'HF_TOKEN' in os.environ}, OPENAI_API_KEY={'OPENAI_API_KEY' in os.environ}")

# Initialize predictor with better error handling
@st.cache_resource
def load_predictor():
    try:
        from predict import Predictor
        predictor = Predictor()
        print("Predictor initialized successfully")
        return predictor
    except Exception as e:
        print(f"Error initializing Predictor: {e}")
        return None

# Default summarization prompt
DEFAULT_SUMMARY_PROMPT = """Please provide a structured summary of the conversation with the following sections:
- **Background:** Briefly describe the context of the discussion.
- **Key Discussion Points:** List the main topics and what was said by each speaker.
- **Decisions Made:** Detail any decisions that were reached.
- **Next Steps:** Outline the action items, who is responsible, and the deadlines."""

# Dictionary of languages for the dropdown, with full names.
LANGUAGES = {
    "No Translation": "None",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Farsi (Persian)": "fa",
    "Arabic": "ar",
    "Greek": "el",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}

def process_audio_file(audio_file, target_language, custom_prompt, predictor):
    """
    Process the uploaded audio file for transcription, translation, and summarization.
    """
    # Check if predictor is available
    if predictor is None:
        return "Error: Predictor not initialized. Please check the logs for initialization errors.", "", "", ""
    
    # Debug: Initial state
    print(f"\n{'='*50}")
    print(f"Processing started at: {datetime.now()}")
    print(f"Audio file: {audio_file}")
    print(f"Target language: {target_language}")
    print(f"Custom prompt provided: {bool(custom_prompt and custom_prompt.strip())}")
    
    if audio_file is None:
        print("ERROR: No audio file provided")
        return "Please upload an audio file first.", "", "", ""
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{audio_file.name}"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Debug: File info
    if os.path.exists(temp_path):
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        print(f"File type: {os.path.splitext(temp_path)[1]}")
    
    try:
        print("\nCalling predictor.predict()...")
        
        result = predictor.predict(
            audio_file_path=temp_path,
            target_language=target_language,
            custom_summary_prompt=custom_prompt
        )
        
        print(f"Predictor returned successfully")
        print(f"Result keys: {list(result.keys())}")
        
        if "error" in result:
            error_message = f"An error occurred: {result['error']}"
            print(f"ERROR in result: {result['error']}")
            return error_message, "", "", ""
        
        # Debug: Process results
        segments = result.get("segments", [])
        print(f"\nNumber of segments: {len(segments)}")
        if segments:
            print(f"First segment: {segments[0]}")
            
        segments_formatted = "\n".join(
            [f"[{s['start']:.2f}-{s['end']:.2f}] {s['speaker']}: {s['text'].strip()}" 
             for s in segments]
        )
        
        language_detected = result.get("language_detected", "N/A")
        print(f"Language detected: {language_detected}")
        
        translation = result.get("translation", "Not requested or an error occurred.")
        summary = result.get("summary", "Not requested or an error occurred.")
        
        print(f"Translation provided: {bool(translation and translation != 'Not requested or an error occurred.')}")
        print(f"Summary provided: {bool(summary and summary != 'Not requested or an error occurred.')}")
        
        print(f"Processing completed at: {datetime.now()}")
        print(f"{'='*50}\n")
        
        return segments_formatted, language_detected, translation, summary
        
    except Exception as e:
        print(f"\nEXCEPTION in process_audio_file: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", "", ""
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Concisio App",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Concisio App")
    st.markdown("Upload an audio file to transcribe, diarize, and optionally translate or summarize.")
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("⚠️ Predictor not initialized. Please check the logs for initialization errors.")
        return
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # Audio upload
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
        )
        
        # Language selection
        target_language = st.selectbox(
            "Translate to (Optional)",
            options=list(LANGUAGES.keys()),
            index=0
        )
        
        # Summary prompt
        custom_prompt = st.text_area(
            "Summarization Prompt (Leave blank to skip)",
            value=DEFAULT_SUMMARY_PROMPT,
            height=200
        )
        
        # Process button
        process_button = st.button(
            "🚀 Process Audio",
            type="primary",
            use_container_width=True,
            disabled=(audio_file is None)
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Transcription & Diarization")
        transcription_placeholder = st.empty()
        
        st.subheader("🌍 Detected Language")
        language_placeholder = st.empty()
    
    with col2:
        st.subheader("🔄 Translation")
        translation_placeholder = st.empty()
        
        st.subheader("📋 Summary")
        summary_placeholder = st.empty()
    
    # Process audio when button is clicked
    if process_button and audio_file is not None:
        with st.spinner("Processing audio... This may take a few minutes."):
            language_code = LANGUAGES.get(target_language, "None")
            
            # Process the audio
            transcription, language_detected, translation, summary = process_audio_file(
                audio_file, language_code, custom_prompt, predictor
            )
            
            # Display results
            with transcription_placeholder.container():
                st.text_area(
                    "Transcription Result",
                    value=transcription,
                    height=300,
                    label_visibility="collapsed"
                )
            
            with language_placeholder.container():
                st.info(f"Detected: {language_detected}")
            
            with translation_placeholder.container():
                st.text_area(
                    "Translation Result",
                    value=translation,
                    height=150,
                    label_visibility="collapsed"
                )
            
            with summary_placeholder.container():
                st.text_area(
                    "Summary Result",
                    value=summary,
                    height=150,
                    label_visibility="collapsed"
                )
            
            st.success("✅ Processing completed!")

if __name__ == "__main__":
    main()
