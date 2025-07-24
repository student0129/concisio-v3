import os
import warnings

# Suppress known warnings first
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*audio is shorter than 30s.*")
warnings.filterwarnings("ignore", message=".*language detection may be inaccurate.*")

# Import torch and related libraries
try:
    import torch
    print(f"Torch imported successfully. Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"Error importing torch: {e}")
    raise e

import whisperx
import openai
from pydub import AudioSegment
from typing import Dict, List, Any, Optional

# Import utils with error handling
try:
    from utils import (
        convert_to_wav,
        merge_text_diarization,
        build_translate_prompt,
        build_summary_prompt
    )
    print("Utils imported successfully")
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    # Define minimal fallback functions
    def convert_to_wav(audio_path):
        return audio_path  # Simple fallback
    
    def merge_text_diarization(segments, diarization):
        return segments  # Simple fallback
    
    def build_translate_prompt(text, lang):
        return f"Translate this to {lang}: {text}"
    
    def build_summary_prompt(text, prompt):
        return f"{prompt}\n\nText: {text}"

class Predictor:
    """
    A class to handle the audio processing pipeline.
    """
    def __init__(self):
        """
        Initializes the Predictor by loading necessary models and setting up the API client.
        """
        print("Starting Predictor initialization...")
        
        # Device detection with extensive error handling
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.compute_type = "float16" if self.device == "cuda" else "int8"
            print(f"Device selected: {self.device}")
            print(f"Compute type: {self.compute_type}")
        except Exception as e:
            print(f"Error in device detection: {e}")
            self.device = "cpu"
            self.compute_type = "int8"
        
        # GPU optimization with error handling
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                
                if gpu_memory < 20:
                    torch.cuda.set_per_process_memory_fraction(0.75)
                    print("Set GPU memory fraction to 75%")
                    
            except Exception as e:
                print(f"GPU optimization error: {e}")
                # Continue without optimization
        
        # Load models
        self._load_whisper_model()
        self._load_diarization_model()
        self._initialize_openai_client()
        
        print("Predictor initialization completed successfully.")
    
    def _load_whisper_model(self):
        """Load Whisper model with progressive fallback"""
        print("Loading Whisper model...")
        
        # Model size options in order of preference
        if self.device == "cuda":
            model_options = [
                ("large-v2", "float16"),
                ("medium", "float16"),
                ("base", "float16"),
                ("base", "int8")  # Last resort
            ]
        else:
            model_options = [
                ("base", "int8"),
                ("tiny", "int8")
            ]
        
        for model_size, compute_type in model_options:
            try:
                print(f"Attempting to load {model_size} model with {compute_type}...")
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                self.model = whisperx.load_model(
                    model_size, 
                    self.device, 
                    compute_type=compute_type,
                    download_root=None
                )
                
                print(f"✅ Successfully loaded {model_size} model")
                return
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg and self.device == "cuda":
                    print(f"❌ GPU memory insufficient for {model_size}, trying smaller model...")
                    continue
                elif "cuda" in error_msg and self.device == "cuda":
                    print(f"❌ CUDA error with {model_size}, trying CPU fallback...")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    try:
                        self.model = whisperx.load_model("base", "cpu", compute_type="int8")
                        print("✅ Successfully loaded base model on CPU")
                        return
                    except Exception as cpu_error:
                        print(f"❌ CPU fallback also failed: {cpu_error}")
                        continue
                else:
                    print(f"❌ Unexpected error with {model_size}: {e}")
                    continue
            except Exception as e:
                print(f"❌ Error loading {model_size}: {e}")
                continue
        
        # If all attempts fail
        raise Exception("❌ Failed to load any Whisper model")
    
    def _load_diarization_model(self):
        """Load diarization model with error handling"""
        print("Loading diarization model...")
        
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("⚠️ Warning: HF_TOKEN not found")
                self.diarize_model = None
                return
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Try different import methods for WhisperX diarization
            try:
                # Method 1: Direct import (newer versions)
                from whisperx.diarize import DiarizationPipeline
                self.diarize_model = DiarizationPipeline(
                    use_auth_token=hf_token, 
                    device=self.device
                )
            except (ImportError, AttributeError):
                # Method 2: Module-level access (older versions)
                try:
                    self.diarize_model = whisperx.diarize.DiarizationPipeline(
                        use_auth_token=hf_token, 
                        device=self.device
                    )
                except AttributeError:
                    # Method 3: Alternative import path
                    import whisperx.diarize
                    self.diarize_model = whisperx.diarize.DiarizationPipeline(
                        use_auth_token=hf_token, 
                        device=self.device
                    )
            
            print("✅ Diarization model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load diarization model: {e}")
            print("Continuing without diarization...")
            self.diarize_model = None
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        print("Initializing OpenAI client...")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("⚠️ Warning: OPENAI_API_KEY not found")
            self.openai_client = None
        else:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: OpenAI client initialization failed: {e}")
                self.openai_client = None

    def predict(
        self, 
        audio_file_path: str, 
        target_language: str = "None", 
        custom_summary_prompt: Optional[str] = None,
        include_diarization: bool = True,
        fast_diarization: bool = False
    ) -> Dict[str, Any]:
        """
        Runs the full prediction pipeline.
        """
        print(f"\n{'='*50}")
        print(f"[Predict] Starting prediction pipeline")
        print(f"[Predict] Audio file: {audio_file_path}")
        print(f"[Predict] Target language: {target_language}")
        print(f"[Predict] Include diarization: {include_diarization}")
        print(f"[Predict] Fast diarization: {fast_diarization}")
        
        output_data = {}
        wav_path = None
        
        try:
            # Convert to WAV
            print("[Predict] Converting to WAV...")
            wav_path = convert_to_wav(audio_file_path)
            print(f"[Predict] WAV path: {wav_path}")
            
            # Load and process audio
            print("[Predict] Loading audio...")
            audio = whisperx.load_audio(wav_path)
            duration = len(audio) / 16000
            print(f"[Predict] Audio loaded, duration: {duration:.1f} seconds")

            # Transcribe
            print("[Predict] Starting transcription...")
            batch_size = 32 if self.device == "cuda" else 8
            
            result = self.model.transcribe(audio, batch_size=batch_size)
            print(f"[Predict] Transcription complete. Language: {result['language']}")
            
            output_data["language_detected"] = result["language"]

            # Align
            print("[Predict] Aligning transcription...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"], 
                    device=self.device
                )
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
                print("[Predict] Alignment complete.")
            except Exception as e:
                print(f"[Predict] Alignment failed: {e}, continuing with unaligned segments...")

            # Process segments with diarization
            final_segments = self._process_diarization(
                audio, result["segments"], include_diarization, fast_diarization
            )
            output_data["segments"] = final_segments

            # Generate full text
            full_text = " ".join([segment['text'].strip() for segment in final_segments])
            print(f"[Predict] Generated full text ({len(full_text)} characters)")

            # Translation and summarization
            self._process_translation_and_summary(
                output_data, full_text, target_language, custom_summary_prompt
            )

            print("[Predict] Pipeline completed successfully")
            return output_data
            
        except Exception as e:
            print(f"[Predict] ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        finally:
            # Cleanup
            if wav_path and os.path.exists(wav_path) and wav_path != audio_file_path:
                try:
                    os.remove(wav_path)
                    print(f"[Predict] Cleaned up: {wav_path}")
                except:
                    pass

    def _process_diarization(self, audio, segments, include_diarization, fast_diarization):
        """Process diarization with error handling"""
        if not include_diarization or self.diarize_model is None:
            print("[Predict] Skipping diarization")
            return [
                {**seg, "speaker": None if not include_diarization else "SPEAKER_00"}
                for seg in segments
            ]
        
        print("[Predict] Processing diarization...")
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Run diarization
            diarize_segments = self.diarize_model(audio)
            
            if diarize_segments.empty:
                print("[Predict] No diarization segments found")
                return [
                    {**seg, "speaker": "SPEAKER_00"}
                    for seg in segments
                ]
            
            print(f"[Predict] Found {len(diarize_segments['speaker'].unique())} speakers")
            
            # Merge with transcription
            merged_segments = merge_text_diarization(segments, diarize_segments)
            return merged_segments
            
        except Exception as e:
            print(f"[Predict] Diarization failed: {e}")
            return [
                {**seg, "speaker": "SPEAKER_00"}
                for seg in segments
            ]

    def _process_translation_and_summary(self, output_data, full_text, target_language, custom_summary_prompt):
        """Handle translation and summarization"""
        # Translation
        if target_language != "None" and output_data["language_detected"] != target_language:
            if self.openai_client is not None:
                try:
                    print(f"[Predict] Translating to {target_language}...")
                    translate_prompt = build_translate_prompt(full_text, target_language)
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a professional translator."},
                            {"role": "user", "content": translate_prompt}
                        ],
                        temperature=0.3,
                    )
                    output_data["translation"] = response.choices[0].message.content
                    print("[Predict] Translation completed")
                except Exception as e:
                    print(f"[Predict] Translation failed: {e}")
                    output_data["translation"] = f"Translation failed: {str(e)}"
            else:
                output_data["translation"] = "Translation not available (OpenAI API key not configured)"
        
        # Summarization
        if custom_summary_prompt and custom_summary_prompt.strip():
            if self.openai_client is not None:
                try:
                    print("[Predict] Generating summary...")
                    summary_prompt = build_summary_prompt(full_text, custom_summary_prompt)
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        temperature=0.5,
                    )
                    output_data["summary"] = response.choices[0].message.content
                    print("[Predict] Summary completed")
                except Exception as e:
                    print(f"[Predict] Summary failed: {e}")
                    output_data["summary"] = f"Summary failed: {str(e)}"
            else:
                output_data["summary"] = "Summary not available (OpenAI API key not configured)"
