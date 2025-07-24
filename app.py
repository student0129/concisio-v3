import os
import gradio as gr
from fastapi import FastAPI
from dotenv import load_dotenv
from predict import Predictor 

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize the predictor
# This loads the models into memory once when the app starts.
predictor = Predictor()

def process_audio(audio_file, task, target_language, custom_prompt):
    """
    Gradio interface function to process the uploaded audio.
    """
    if audio_file is None:
        # Return empty strings for all outputs if no file is uploaded
        return "Please upload an audio file first.", "", "", ""

    # The Gradio audio component saves the file to a temp location,
    # and the 'audio_file' variable holds the path to it.
    result = predictor.predict(
        audio_file_path=audio_file,
        task=task,
        target_language=target_language,
        custom_summary_prompt=custom_prompt
    )

    # Handle potential errors from the prediction pipeline
    if "error" in result:
        error_message = f"An error occurred: {result['error']}"
        return error_message, "", "", ""

    # Format the output for Gradio UI
    segments_formatted = "\n".join(
        [f"[{s['start']:.2f}-{s['end']:.2f}] {s['speaker']}: {s['text'].strip()}" for s in result.get("segments", [])]
    )
    language_detected = result.get("language_detected", "N/A")
    translation = result.get("translation", "Not requested or an error occurred.")
    summary = result.get("summary", "Not requested or an error occurred.")

    return segments_formatted, language_detected, translation, summary

# Create the Gradio Interface using gr.Blocks for more control
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# WhisperX GPT App")
    gr.Markdown("Upload an audio file to transcribe, diarize, and optionally translate or summarize.")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            
            gr.Markdown("### Options")
            task_selection = gr.Radio(
                ["transcribe", "translate"], label="Task", value="transcribe",
                info="Choose 'translate' to translate the transcription to the target language."
            )
            language_dropdown = gr.Dropdown(
                choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar"], 
                label="Target Language (for Translation)",
                value="en"
            )
            summary_prompt_input = gr.Textbox(
                label="Summarization Prompt",
                info="Provide a prompt to summarize the text. Leave blank to skip.",
                placeholder="e.g., Summarize the key decisions from this meeting."
            )
            submit_button = gr.Button("Process Audio", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Results")
            detected_language_output = gr.Textbox(label="Detected Language", interactive=False)
            transcription_output = gr.Textbox(label="Transcription & Diarization", lines=15, interactive=False)
            translation_output = gr.Textbox(label="Translation", lines=5, interactive=False)
            summary_output = gr.Textbox(label="Summary", lines=5, interactive=False)
    
    # Connect the button to the processing function
    submit_button.click(
        fn=process_audio,
        inputs=[audio_input, task_selection, language_dropdown, summary_prompt_input],
        outputs=[transcription_output, detected_language_output, translation_output, summary_output],
    )

# Mount the Gradio app to the FastAPI application
app = gr.mount_gradio_app(app, interface, path="/")

if __name__ == "__main__":
    import uvicorn
    # This allows running the app directly with 'python app.py' for local development
    uvicorn.run(app, host="0.0.0.0", port=7860)
