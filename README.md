# WhisperX GPT App

This web application provides a user-friendly interface to transcribe, diarize, translate, and summarize audio files using a powerful pipeline of WhisperX and OpenAI's GPT-4o.

## Core Features

* **Audio Transcription & Diarization**: Upload audio files (MP3, WAV, M4A, etc.) and receive a full transcription with speaker labels.
* **Multi-language Support**: Automatically detects the language of the audio.
* **Optional Translation**: Translate the transcription to a selected target language using GPT-4o.
* **Optional Summarization**: Generate a summary of the transcription using a default or a custom user-provided prompt with GPT-4o.
* **Simple Web Interface**: Built with Gradio and FastAPI for ease of use.

## Project Structure

```
whisperx-gpt-space/
├── app.py                # FastAPI + Gradio entry point
├── predict.py            # Core logic pipeline (Predictor class)
├── utils.py              # Helper functions (audio conversion, prompts, etc.)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .env                  # Local environment variables (create this yourself)
├── .gitignore            # To exclude virtual environments and cache
└── uploads/              # Temporary directory for audio files (created at runtime)
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/student0129/concisio-v3](https://github.com/student0129/concisio-v3)
cd concisio-v3
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Python 3
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required packages from `requirements.txt`.

> **Note:** This step can take a while as it downloads the machine learning models. Ensure you have a stable internet connection. `torch` installation might vary based on your system (CPU/GPU). The command below should work for most setups.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

You will need an API key from OpenAI to use the translation and summarization features.

Create a file named `.env` in the root of the project directory and add your key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

If you are deploying to Hugging Face Spaces and need to use a private diarization model, you may also need a Hugging Face token:

```
HF_TOKEN=your_hugging_face_token_here
```

## Running the Application Locally

Once the setup is complete, you can run the application using Uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 7860
```

Open your web browser and navigate to `http://127.0.0.1:7860`.

## Deployment

### Hugging Face Spaces (Recommended)

1. **Create a new Space**: Go to Hugging Face and create a new Gradio SDK Space.
2. **Link your GitHub Repo**: Connect the Space to your GitHub repository.
3. **Add Secrets**: In the Space settings, go to the "Secrets" section and add your `OPENAI_API_KEY`. This is crucial for the app to function. If you are using a private diarization model, add your `HF_TOKEN` as well.
4. **Deploy**: The Space will automatically pull your code from the `main` branch and deploy the application. Any push to the `main` branch will trigger a new deployment.
