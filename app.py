import os
import gradio as gr
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Add debugging
print("Starting Concisio App...")
print(f"Environment variables loaded: HF_TOKEN={'HF_TOKEN' in os.environ}, OPENAI_API_KEY={'OPENAI_API_KEY' in os.environ}")

try:
    from predict import Predictor
    predictor = Predictor()
    print("Predictor initialized successfully")
except Exception as e:
    print(f"Error initializing Predictor: {e}")
    predictor = None

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

def process_audio_file(audio_file, target_language, custom_prompt):
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
    
    # Debug: File info
    if os.path.exists(audio_file):
        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        print(f"File type: {os.path.splitext(audio_file)[1]}")
    
    try:
        print("\nCalling predictor.predict()...")
        
        result = predictor.predict(
            audio_file_path=audio_file,
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

def gradio_interface(audio, language_name, prompt):
    """
    Wrapper function for Gradio interface.
    """
    language_code = LANGUAGES.get(language_name, "None")
    return process_audio_file(audio, language_code, prompt)

# Create the Gradio Interface using Gradio 3.x API
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(
            choices=list(LANGUAGES.keys()), 
            value="No Translation",
            label="Translate to (Optional)"
        ),
        gr.Textbox(
            label="Summarization Prompt (Leave blank to skip)",
            value=DEFAULT_SUMMARY_PROMPT,
            lines=8
        )
    ],
    outputs=[
        gr.Textbox(label="Transcription & Diarization", lines=15),
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Translation", lines=5),
        gr.Textbox(label="Summary", lines=5)
    ],
    title="Concisio App",
    description="Upload an audio file to transcribe, diarize, and optionally translate or summarize.",
    theme="default"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
