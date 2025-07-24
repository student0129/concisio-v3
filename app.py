import os
import gradio as gr
from dotenv import load_dotenv
from predict import Predictor 

# Load environment variables from .env file
load_dotenv()

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

# Initialize the predictor
predictor = Predictor()

def process_audio_file(audio_file, target_language, custom_prompt):
    """
    Process the uploaded audio file for transcription, translation, and summarization.
    """
    if audio_file is None:
        return "Please upload an audio file first.", "", "", ""

    result = predictor.predict(
        audio_file_path=audio_file,
        target_language=target_language,
        custom_summary_prompt=custom_prompt
    )

    if "error" in result:
        error_message = f"An error occurred: {result['error']}"
        return error_message, "", "", ""

    segments_formatted = "\n".join(
        [f"[{s['start']:.2f}-{s['end']:.2f}] {s['speaker']}: {s['text'].strip()}" for s in result.get("segments", [])]
    )
    language_detected = result.get("language_detected", "N/A")
    translation = result.get("translation", "Not requested or an error occurred.")
    summary = result.get("summary", "Not requested or an error occurred.")

    return segments_formatted, language_detected, translation, summary

def gradio_interface(audio, language_name, prompt):
    """
    Wrapper function for Gradio interface.
    """
    language_code = LANGUAGES.get(language_name, "None")
    return process_audio_file(audio, language_code, prompt)

# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Concisio App")
    gr.Markdown("Upload an audio file to transcribe, diarize, and optionally translate or summarize.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            
            gr.Markdown("### Options")
            language_dropdown = gr.Dropdown(
                choices=list(LANGUAGES.keys()), 
                value="No Translation",
                label="Translate to (Optional)",
            )
            summary_prompt_input = gr.Textbox(
                label="Summarization Prompt (Leave blank to skip)",
                value=DEFAULT_SUMMARY_PROMPT,
                lines=8,
            )
            submit_button = gr.Button("Process Audio", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Results")
            detected_language_output = gr.Textbox(label="Detected Language", interactive=False)
            transcription_output = gr.Textbox(label="Transcription & Diarization", lines=15, interactive=False)
            translation_output = gr.Textbox(label="Translation", lines=5, interactive=False)
            summary_output = gr.Textbox(label="Summary", lines=5, interactive=False)

    # Set up the click event with a unique API name
    submit_button.click(
        fn=gradio_interface,
        inputs=[audio_input, language_dropdown, summary_prompt_input],
        outputs=[transcription_output, detected_language_output, translation_output, summary_output],
        api_name="audio_processing"
    )

# For Hugging Face Spaces, you might want to use just Gradio without FastAPI
if __name__ == "__main__":
    demo.launch()
