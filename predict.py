import os
import torch
import whisperx
import openai
from pydub import AudioSegment
from utils import (
    convert_to_wav,
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"Initializing Predictor on {self.device}")

        # Use smaller model if on CPU for better performance
        model_size = "large-v2" if self.device == "cuda" else "base"
        
        self.model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type)
        self.diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=self.device)
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def predict(self, audio_file_path, target_language="None", custom_summary_prompt=None):
        """
        Runs the full prediction pipeline.
        The 'task' parameter has been removed.
        """
        output_data = {}
        wav_path = None
        
        try:
            wav_path = convert_to_wav(audio_file_path)
            audio = whisperx.load_audio(wav_path)

            result = self.model.transcribe(audio, batch_size=16)
            output_data["language_detected"] = result["language"]

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

            diarize_segments = self.diarize_model(audio)
            
            final_segments = merge_text_diarization(result["segments"], diarize_segments)
            output_data["segments"] = final_segments

            full_text = " ".join([segment['text'].strip() for segment in final_segments])

            # Logic is updated to check if a valid language was selected from the dropdown.
            if target_language != "None" and output_data["language_detected"] != target_language:
                translate_prompt = build_translate_prompt(full_text, target_language)
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a professional translator."},
                              {"role": "user", "content": translate_prompt}],
                    temperature=0.3,
                )
                output_data["translation"] = response.choices[0].message.content
            
            if custom_summary_prompt and custom_summary_prompt.strip():
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
            return {"error": str(e)}
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        return output_data
