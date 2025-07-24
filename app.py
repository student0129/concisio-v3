import os
import gradio as gr
from fastapi import FastAPI
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

# Initialize FastAPI app
app = FastAPI()

# Initialize the predictor
predictor = Predictor()

def process_audio(audio_file, target_language, custom_prompt):
    """
    Gradio interface function to process the uploaded audio.
    Transcription is now always performed. Translation is optional.
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

# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as interface:
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
    
    # --- API ERROR FIX ---
    # Using a dedicated wrapper function instead of a lambda to ensure Gradio's
    # API generator can correctly parse the function and its inputs.
    def gradio_wrapper(audio, lang_name, prompt):
        """
        A wrapper function to handle the inputs from the Gradio interface
        and pass them to the main processing function.
        """
        lang_code = LANGUAGES.get(lang_name, "None")
        return process_audio(audio, lang_code, prompt)

    submit_button.click(
        fn=gradio_wrapper, # Use the named wrapper function
        inputs=[audio_input, language_dropdown, summary_prompt_input],
        outputs=[transcription_output, detected_language_output, translation_output, summary_output],
        api_name="process_audio" 
    )

app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
