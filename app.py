import gradio as gr

def simple_echo(text):
    return f"Echo: {text}"

def mock_audio_process(audio, language, prompt):
    if audio is None:
        return "No audio file provided", "N/A", "No translation", "No summary"
    return "Mock transcription result", "English", "Mock translation", "Mock summary"

# Create test interface
iface = gr.Interface(
    fn=mock_audio_process,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File"),
        gr.Dropdown(
            choices=["No Translation", "English", "Spanish"], 
            value="No Translation",
            label="Translate to (Optional)"
        ),
        gr.Textbox(
            label="Summarization Prompt",
            value="Test prompt",
            lines=3
        )
    ],
    outputs=[
        gr.Textbox(label="Transcription & Diarization", lines=10),
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="Translation", lines=3),
        gr.Textbox(label="Summary", lines=3)
    ],
    title="Test App - No ML Libraries",
    description="Testing if Gradio works without ML dependencies."
)

if __name__ == "__main__":
    iface.launch()
