import os
import torch
import whisperx
import openai
from pydub import AudioSegment
from typing import Dict, List, Any, Optional
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

        print(f"Initializing Predictor on {self.device} with {self.compute_type} compute type")

        # Use smaller model if on CPU for better performance
        model_size = "large-v2" if self.device == "cuda" else "base"
        
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisperx.load_model(model_size, self.device, compute_type=self.compute_type)
        print("Whisper model loaded.")
        
        # Initialize diarization model with better error handling
        print("Loading Diarization model...")
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("Warning: HF_TOKEN not found. Diarization may not work properly.")
            
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, 
                device=self.device
            )
            print("Diarization model loaded.")
        except Exception as e:
            print(f"Warning: Could not load diarization model: {e}")
            self.diarize_model = None

        # Initialize OpenAI client
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("Warning: OPENAI_API_KEY not found. Translation and summarization will not work.")
            self.openai_client = None
        else:
            self.openai_client = openai.OpenAI(api_key=openai_key)
            
        print("Predictor initialized successfully.")

    def predict(
        self, 
        audio_file_path: str, 
        target_language: str = "None", 
        custom_summary_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the full prediction pipeline.
        
        Args:
            audio_file_path: Path to the audio file
            target_language: Target language code for translation
            custom_summary_prompt: Custom prompt for summarization
            
        Returns:
            Dictionary containing the results or error information
        """

        print(f"\n[Predict] Starting prediction pipeline")
        print(f"[Predict] Audio file: {audio_file_path}")
        print(f"[Predict] Target language: {target_language}")
    
        output_data = {}
        wav_path = None
        
        try:
            # Convert to WAV
            print("[Predict] Converting to WAV...")
            wav_path = convert_to_wav(audio_file_path)
            print(f"[Predict] WAV path: {wav_path}")
            
            # Load audio
            print("[Predict] Loading audio...")
            audio = whisperx.load_audio(wav_path)
            print(f"[Predict] Audio loaded, duration: {len(audio)/16000:.1f} seconds")

            # Transcribe
            print("[Predict] Starting transcription...")
            result = self.model.transcribe(audio, batch_size=16)
            print(f"[Predict] Transcription complete. Language: {result['language']}")
            if result['segments']:
                print(f"[Predict]  > Sample segment: {result['segments'][0]['text']}")
        
            output_data["language_detected"] = result["language"]

            # Align
            print("[Predict] Aligning transcription...")
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            print("[Predict] Alignment complete.")
            if result['segments']:
                 print(f"[Predict]  > Sample aligned segment: {result['segments'][0]['text']}")

            # Diarize (with fallback if model not available)
            final_segments = []
            if self.diarize_model is not None:
                print("[Predict] Performing speaker diarization...")
                try:
                    diarize_segments = self.diarize_model(audio)
                    print("[Predict] Diarization complete.")
                    if not diarize_segments.empty:
                        print(f"[Predict]  > Found {len(diarize_segments['speaker'].unique())} speakers. Sample diarization: {diarize_segments.iloc[0].to_dict()}")
                    
                    # Merge
                    print("[Predict] Merging transcription and diarization...")
                    final_segments = merge_text_diarization(result["segments"], diarize_segments)
                    print("[Predict] Merging complete.")
                except Exception as e:
                    print(f"[Predict] Diarization failed: {e}. Using transcription only.")
                    final_segments = [
                        {
                            **seg,
                            "speaker": "SPEAKER_00"
                        }
                        for seg in result["segments"]
                    ]
            else:
                print("[Predict] Diarization model not available. Using transcription only.")
                final_segments = [
                    {
                        **seg,
                        "speaker": "SPEAKER_00"
                    }
                    for seg in result["segments"]
                ]
            
            output_data["segments"] = final_segments
            if final_segments:
                print(f"[Predict]  > Sample final segment: {final_segments[0]}")

            full_text = " ".join([segment['text'].strip() for segment in final_segments])
            print(f"[Predict] Generated full text for API calls (first 100 chars): '{full_text[:100]}...'")

            # Translate
            if target_language != "None" and output_data["language_detected"] != target_language:
                if self.openai_client is not None:
                    print(f"[Predict] Translating text to {target_language}...")
                    translate_prompt = build_translate_prompt(full_text, target_language)
                    print(f"[Predict]  > OpenAI Translate Prompt: '{translate_prompt[:150]}...'")
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "You are a professional translator."},
                                  {"role": "user", "content": translate_prompt}],
                        temperature=0.3,
                    )
                    output_data["translation"] = response.choices[0].message.content
                    print("[Predict] Translation complete.")
                else:
                    output_data["translation"] = "Translation not available (OpenAI API key not configured)"
                    print("[Predict] Translation skipped - no OpenAI API key")
            
            # Summarize
            if custom_summary_prompt and custom_summary_prompt.strip():
                if self.openai_client is not None:
                    print("[Predict] Summarizing text...")
                    summary_prompt = build_summary_prompt(full_text, custom_summary_prompt)
                    print(f"[Predict]  > OpenAI Summary Prompt: '{summary_prompt[:150]}...'")
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "You are a helpful assistant that summarizes text."},
                                  {"role": "user", "content": summary_prompt}],
                        temperature=0.5,
                    )
                    output_data["summary"] = response.choices[0].message.content
                    print("[Predict] Summarization complete.")
                else:
                    output_data["summary"] = "Summarization not available (OpenAI API key not configured)"
                    print("[Predict] Summarization skipped - no OpenAI API key")

        except Exception as e:
            print(f"[Predict] ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                print(f"[Predict] Cleaned up temporary file: {wav_path}")
        
        print("[Predict] Prediction pipeline finished successfully.")
        return output_data
