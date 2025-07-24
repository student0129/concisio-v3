import os
import torch
import whisperx
import openai
from pydub import AudioSegment
from utils import (
    convert_to_wav, 
    split_chunks, 
    merge_text_diarization,
    build_translate_prompt,
    build_summary_prompt
)

class Predictor:
    """
    A class to handle the audio processing pipeline.
    """
    def __init__(self):
        """
        Initializes the Predictor by loading necessary models and setting up the API client.
        """
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Load WhisperX model
        # Using "large-v2" for best accuracy, but can be changed to smaller models
        # e.g., "base", "small", "medium" for faster processing.
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

        # Load Diarization model
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=self.device)

        # Setup OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def predict(self, audio_file_path, task="transcribe", target_language="en", custom_summary_prompt=None):
        """
        Runs the full prediction pipeline.
        
        Args:
            audio_file_path (str): Path to the input audio file.
            task (str): "transcribe" or "translate".
            target_language (str): Target language code for translation.
            custom_summary_prompt (str, optional): A custom prompt for summarization.

        Returns:
            dict: A dictionary containing the results.
        """
        output_data = {}
        
        try:
            # 1. Convert audio to 16kHz mono WAV
            wav_path = convert_to_wav(audio_file_path)
            audio = whisperx.load_audio(wav_path)

            # 2. Transcribe the audio
            # The model automatically detects the language
            result = self.model.transcribe(audio, batch_size=16)
            output_data["language_detected"] = result["language"]

            # 3. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            # 4. Diarize the audio
            diarize_segments = self.diarize_model(audio)
            
            # 5. Merge transcription and diarization
            final_segments = merge_text_diarization(result["segments"], diarize_segments)
            output_data["segments"] = final_segments

            # Get the full text for translation/summarization
            full_text = " ".join([segment['text'] for segment in final_segments])

            # 6. Translate if requested
            if task == "translate" and output_data["language_detected"] != target_language:
                translate_prompt = build_translate_prompt(full_text, target_language)
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a professional translator."},
                              {"role": "user", "content": translate_prompt}],
                    temperature=0.3,
                )
                output_data["translation"] = response.choices[0].message.content
            
            # 7. Summarize if a prompt is provided (default or custom)
            if custom_summary_prompt:
                summary_prompt = build_summary_prompt(full_text, custom_summary_prompt)
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a helpful assistant that summarizes text."},
                              {"role": "user", "content": summary_prompt}],
                    temperature=0.5,
                )
                output_data["summary"] = response.choices[0].message.content

        except Exception as e:
            print(f"An error occurred: {e}")
            # Return a partial result or an error message
            return {"error": str(e)}
        finally:
            # Clean up temporary WAV file
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)

        return output_data
