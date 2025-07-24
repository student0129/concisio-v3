import os
import torch  # Import torch at module level
import whisperx
import openai
from pydub import AudioSegment
from typing import Dict, List, Any, Optional
import warnings

# Suppress known warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

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
        # Enhanced GPU detection and optimization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"Initializing Predictor on {self.device} with {self.compute_type} compute type")
        
        # GPU memory optimization and error handling
        if self.device == "cuda":
            try:
                # Clear any existing GPU memory
                torch.cuda.empty_cache()
                
                # Set memory management
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                
                # Set memory fraction to prevent OOM for T4
                if gpu_memory < 20:  # For T4 (16GB)
                    torch.cuda.set_per_process_memory_fraction(0.75)
                    print("Set GPU memory fraction to 75% for T4 optimization")
                    
            except Exception as e:
                print(f"GPU setup warning: {e}")
                # Don't fallback to CPU immediately, try to continue
        
        # Load models with fallback strategy
        self._load_whisper_model()
        self._load_diarization_model()
        self._initialize_openai_client()
        
        print("Predictor initialized successfully.")
    
    def _load_whisper_model(self):
        """Load Whisper model with GPU memory fallback"""
        model_attempts = [
            ("large-v2", "float16") if self.device == "cuda" else ("base", "int8"),
            ("medium", "float16") if self.device == "cuda" else ("base", "int8"),
            ("base", "float16") if self.device == "cuda" else ("base", "int8"),
        ]
        
        for model_size, compute_type in model_attempts:
            try:
                print(f"Loading Whisper model: {model_size} with {compute_type}...")
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()  # Clear memory before loading
                
                self.model = whisperx.load_model(
                    model_size, 
                    self.device, 
                    compute_type=compute_type,
                    download_root=None,
                    threads=4 if self.device == "cpu" else None
                )
                print(f"Whisper model {model_size} loaded successfully.")
                return
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    print(f"GPU memory insufficient for {model_size}, trying smaller model...")
                    continue
                else:
                    print(f"Error loading {model_size}: {e}")
                    if model_size == "base":  # Last attempt failed
                        # Try CPU as final fallback
                        print("Switching to CPU mode...")
                        self.device = "cpu"
                        self.compute_type = "int8"
                        self.model = whisperx.load_model("base", "cpu", compute_type="int8")
                        return
            except Exception as e:
                print(f"Unexpected error loading {model_size}: {e}")
                continue
        
        raise Exception("Failed to load any Whisper model")
    
    def _load_diarization_model(self):
        """Load diarization model with error handling"""
        print("Loading Diarization model...")
        try:
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print("Warning: HF_TOKEN not found. Diarization may not work properly.")
            
            # Clear GPU memory before loading diarization model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, 
                device=self.device,
                cache_dir=None,
            )
            print("Diarization model loaded successfully.")
            
        except Exception as e:
            print(f"Warning: Could not load diarization model: {e}")
            print("Continuing without diarization capability...")
            self.diarize_model = None
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client"""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("Warning: OPENAI_API_KEY not found. Translation and summarization will not work.")
            self.openai_client = None
        else:
            self.openai_client = openai.OpenAI(api_key=openai_key)

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
        
        Args:
            audio_file_path: Path to the audio file
            target_language: Target language code for translation
            custom_summary_prompt: Custom prompt for summarization
            include_diarization: Whether to perform speaker diarization
            fast_diarization: Whether to use fast diarization mode
            
        Returns:
            Dictionary containing the results or error information
        """

        print(f"\n[Predict] Starting prediction pipeline")
        print(f"[Predict] Audio file: {audio_file_path}")
        print(f"[Predict] Target language: {target_language}")
        print(f"[Predict] Include diarization: {include_diarization}")
        print(f"[Predict] Custom prompt provided: {bool(custom_summary_prompt and custom_summary_prompt.strip())}")
    
        output_data = {}
        wav_path = None
        
        try:
            # Convert to WAV with optimization
            print("[Predict] Converting to WAV...")
            wav_path = convert_to_wav(audio_file_path)
            print(f"[Predict] WAV path: {wav_path}")
            
            # Check if file should be chunked for large files (>15 minutes)
            audio_info = whisperx.load_audio(wav_path)
            duration_minutes = len(audio_info) / (16000 * 60)
            
            if duration_minutes > 15:
                print(f"[Predict] Large file detected ({duration_minutes:.1f} minutes). Using chunked processing...")
                return self._process_large_file(wav_path, target_language, custom_summary_prompt, include_diarization, fast_diarization)
            
            # Regular processing for smaller files
            return self._process_regular_file(wav_path, target_language, custom_summary_prompt, include_diarization, fast_diarization)

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

    def _process_regular_file(self, wav_path, target_language, custom_summary_prompt, include_diarization, fast_diarization):
        """Process regular-sized audio files (under 15 minutes)"""
        output_data = {}
        
        # Load audio with optimization
        print("[Predict] Loading audio...")
        audio = whisperx.load_audio(wav_path)
        print(f"[Predict] Audio loaded, duration: {len(audio)/16000:.1f} seconds")

        # Transcribe with optimized batch size
        print("[Predict] Starting transcription...")
        batch_size = 32 if self.device == "cuda" else 8
        result = self.model.transcribe(audio, batch_size=batch_size)
        print(f"[Predict] Transcription complete. Language: {result['language']}")
        
        output_data["language_detected"] = result["language"]

        # Align with device-specific optimization
        print("[Predict] Aligning transcription...")
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

        # Process diarization
        final_segments = self._process_diarization(audio, result["segments"], include_diarization, fast_diarization)
        output_data["segments"] = final_segments

        # Process translation and summarization
        full_text = " ".join([segment['text'].strip() for segment in final_segments])
        self._process_translation_and_summary(output_data, full_text, target_language, custom_summary_prompt)
        
        return output_data

    def _process_large_file(self, wav_path, target_language, custom_summary_prompt, include_diarization, fast_diarization):
        """Process large audio files using chunking for better performance"""
        from utils import chunk_audio_for_processing, merge_chunked_results, cleanup_temp_files
        
        # Split into chunks
        chunk_paths = chunk_audio_for_processing(wav_path, chunk_duration_minutes=10)
        chunk_results = []
        
        try:
            for i, chunk_path in enumerate(chunk_paths):
                print(f"[Predict] Processing chunk {i+1}/{len(chunk_paths)}...")
                
                # Process each chunk
                chunk_result = self._process_regular_file(
                    chunk_path, target_language, None, include_diarization, fast_diarization
                )
                chunk_results.append(chunk_result)
            
            # Merge chunk results
            merged_result = merge_chunked_results(chunk_results)
            
            # Process translation and summarization on merged text
            if chunk_results:
                full_text = " ".join([segment['text'].strip() for segment in merged_result["segments"]])
                self._process_translation_and_summary(merged_result, full_text, target_language, custom_summary_prompt)
            
            return merged_result
            
        finally:
            # Cleanup chunk files
            cleanup_temp_files(*chunk_paths)

    def _process_diarization(self, audio, segments, include_diarization, fast_diarization):
        """Handle diarization processing with optimizations"""
        from utils import merge_text_diarization
        
        if include_diarization and self.diarize_model is not None:
            print("[Predict] Performing speaker diarization...")
            try:
                if fast_diarization:
                    print("[Predict] Using fast diarization mode...")
                    diarize_segments = self.diarize_model(
                        audio,
                        min_speakers=2,
                        max_speakers=6,
                    )
                else:
                    print("[Predict] Using standard diarization mode...")
                    diarize_segments = self.diarize_model(
                        audio,
                        min_speakers=2,
                        max_speakers=10,
                    )
                
                print("[Predict] Diarization complete.")
                if not diarize_segments.empty:
                    print(f"[Predict]  > Found {len(diarize_segments['speaker'].unique())} speakers")
                
                return merge_text_diarization(segments, diarize_segments)
                
            except Exception as e:
                print(f"[Predict] Diarization failed: {e}. Using transcription only.")
                return [
                    {**seg, "speaker": "SPEAKER_00"}
                    for seg in segments
                ]
        else:
            if not include_diarization:
                print("[Predict] Diarization disabled by user.")
            else:
                print("[Predict] Diarization model not available.")
            
            return [
                {**seg, "speaker": "SPEAKER_00" if include_diarization else None}
                for seg in segments
            ]

    def _process_translation_and_summary(self, output_data, full_text, target_language, custom_summary_prompt):
        """Handle translation and summarization"""
        from utils import build_translate_prompt, build_summary_prompt
        
        # Translate
        if target_language != "None" and output_data["language_detected"] != target_language:
            if self.openai_client is not None:
                print(f"[Predict] Translating text to {target_language}...")
                translate_prompt = build_translate_prompt(full_text, target_language)
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
        
        # Summarize
        if custom_summary_prompt and custom_summary_prompt.strip():
            if self.openai_client is not None:
                print("[Predict] Summarizing text...")
                summary_prompt = build_summary_prompt(full_text, custom_summary_prompt)
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
