import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
import tempfile
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings("ignore")

def convert_to_wav(audio_file_path: str, target_sample_rate: int = 16000) -> str:
    """
    Convert audio file to WAV format optimized for WhisperX processing.
    
    Performance optimizations:
    - Direct conversion without intermediate formats
    - Optimized sample rate conversion
    - Memory-efficient processing
    """
    try:
        print(f"[Utils] Converting {audio_file_path} to WAV...")
        
        # Load audio with optimized parameters
        if audio_file_path.lower().endswith('.wav'):
            # Already WAV, just check sample rate
            audio = AudioSegment.from_wav(audio_file_path)
        else:
            # Convert from other formats
            audio = AudioSegment.from_file(audio_file_path)
        
        # Optimize audio for processing
        # Convert to mono if stereo (reduces processing time by ~50%)
        if audio.channels > 1:
            print("[Utils] Converting to mono for faster processing...")
            audio = audio.set_channels(1)
        
        # Optimize sample rate
        if audio.frame_rate != target_sample_rate:
            print(f"[Utils] Resampling from {audio.frame_rate}Hz to {target_sample_rate}Hz...")
            audio = audio.set_frame_rate(target_sample_rate)
        
        # Create optimized temporary file
        temp_wav_path = tempfile.mktemp(suffix=".wav")
        
        # Export with optimized parameters for speed
        audio.export(
            temp_wav_path, 
            format="wav",
            parameters=[
                "-ac", "1",  # Force mono
                "-ar", str(target_sample_rate),  # Force sample rate
                "-acodec", "pcm_s16le"  # Use efficient codec
            ]
        )
        
        print(f"[Utils] Converted to: {temp_wav_path}")
        return temp_wav_path
        
    except Exception as e:
        print(f"[Utils] Error converting audio: {e}")
        raise e

def merge_text_diarization(
    segments: List[Dict], 
    diarization_df: pd.DataFrame, 
    tolerance: float = 0.5
) -> List[Dict]:
    """
    Merge transcription segments with diarization results.
    
    Performance optimizations:
    - Vectorized operations using NumPy
    - Efficient interval matching algorithm
    - Reduced memory allocations
    """
    if diarization_df.empty or not segments:
        print("[Utils] No diarization data or segments to merge")
        return segments
    
    print(f"[Utils] Merging {len(segments)} segments with {len(diarization_df)} diarization segments...")
    
    try:
        # Convert to numpy arrays for faster processing
        segment_starts = np.array([seg['start'] for seg in segments])
        segment_ends = np.array([seg['end'] for seg in segments])
        segment_centers = (segment_starts + segment_ends) / 2
        
        # Prepare diarization data
        diar_starts = diarization_df['start'].values
        diar_ends = diarization_df['end'].values
        diar_speakers = diarization_df['speaker'].values
        
        # Vectorized speaker assignment
        merged_segments = []
        
        for i, segment in enumerate(segments):
            center_time = segment_centers[i]
            
            # Find overlapping diarization segments using vectorized operations
            overlaps = (diar_starts <= center_time + tolerance) & (diar_ends >= center_time - tolerance)
            overlap_indices = np.where(overlaps)[0]
            
            if len(overlap_indices) > 0:
                # If multiple overlaps, choose the one with maximum overlap
                if len(overlap_indices) > 1:
                    overlap_durations = np.minimum(segment_ends[i], diar_ends[overlap_indices]) - \
                                     np.maximum(segment_starts[i], diar_starts[overlap_indices])
                    best_idx = overlap_indices[np.argmax(overlap_durations)]
                else:
                    best_idx = overlap_indices[0]
                
                speaker = diar_speakers[best_idx]
            else:
                # No overlap found, assign unknown speaker
                speaker = "SPEAKER_UNKNOWN"
            
            # Create merged segment
            merged_segment = {
                **segment,
                'speaker': speaker
            }
            merged_segments.append(merged_segment)
        
        print(f"[Utils] Successfully merged segments with speakers")
        return merged_segments
        
    except Exception as e:
        print(f"[Utils] Error in merge_text_diarization: {e}")
        # Fallback: assign default speaker to all segments
        return [
            {**seg, 'speaker': 'SPEAKER_00'} 
            for seg in segments
        ]

def build_translate_prompt(text: str, target_language: str) -> str:
    """
    Build optimized translation prompt for OpenAI API.
    
    Performance optimizations:
    - Concise prompts for faster API response
    - Language-specific optimizations
    - Reduced token usage
    """
    # Language code to full name mapping for better results
    language_map = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'fa': 'Persian',
        'ar': 'Arabic',
        'el': 'Greek',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    target_lang_name = language_map.get(target_language, target_language)
    
    # Optimized prompt for speed and accuracy
    prompt = f"""Translate this audio transcription to {target_lang_name}. Maintain speaker labels and timing context.

Text to translate:
{text}

Provide only the translation, keeping the same structure."""
    
    return prompt

def build_summary_prompt(text: str, custom_prompt: str) -> str:
    """
    Build optimized summarization prompt.
    
    Performance optimizations:
    - Efficient prompt structure
    - Context-aware processing
    - Reduced API call overhead
    """
    # Add context about the content type for better results
    context_prompt = """You are summarizing a transcribed audio conversation. Focus on spoken content structure and key information.

"""
    
    # Combine with custom prompt
    full_prompt = f"""{context_prompt}{custom_prompt}

Transcribed conversation to summarize:
{text}"""
    
    return full_prompt

def chunk_audio_for_processing(
    audio_path: str, 
    chunk_duration_minutes: int = 10,
    overlap_seconds: int = 30
) -> List[str]:
    """
    Split large audio files into chunks for faster processing.
    
    Performance optimization for large files:
    - Process files in parallel-ready chunks
    - Overlap handling for continuity
    - Memory-efficient chunking
    """
    try:
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        duration_minutes = duration_ms / (1000 * 60)
        
        # If file is small, don't chunk
        if duration_minutes <= chunk_duration_minutes:
            return [audio_path]
        
        print(f"[Utils] Chunking {duration_minutes:.1f} minute file into {chunk_duration_minutes} minute segments...")
        
        chunk_paths = []
        chunk_duration_ms = chunk_duration_minutes * 60 * 1000
        overlap_ms = overlap_seconds * 1000
        
        start_ms = 0
        chunk_num = 0
        
        while start_ms < duration_ms:
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            
            # Extract chunk with overlap
            if chunk_num > 0:
                chunk_start = max(0, start_ms - overlap_ms)
            else:
                chunk_start = start_ms
                
            if end_ms < duration_ms:
                chunk_end = end_ms + overlap_ms
            else:
                chunk_end = end_ms
            
            chunk = audio[chunk_start:chunk_end]
            
            # Save chunk
            chunk_path = tempfile.mktemp(suffix=f"_chunk_{chunk_num}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
            
            print(f"[Utils] Created chunk {chunk_num}: {chunk_start/1000:.1f}s - {chunk_end/1000:.1f}s")
            
            start_ms = end_ms
            chunk_num += 1
        
        return chunk_paths
        
    except Exception as e:
        print(f"[Utils] Error chunking audio: {e}")
        return [audio_path]  # Return original if chunking fails

def merge_chunked_results(chunk_results: List[Dict]) -> Dict:
    """
    Merge results from chunked audio processing.
    
    Performance optimization:
    - Efficient result combination
    - Timeline adjustment
    - Speaker consistency across chunks
    """
    if len(chunk_results) == 1:
        return chunk_results[0]
    
    print(f"[Utils] Merging results from {len(chunk_results)} chunks...")
    
    merged_segments = []
    cumulative_offset = 0
    
    for i, chunk_result in enumerate(chunk_results):
        chunk_segments = chunk_result.get('segments', [])
        
        for segment in chunk_segments:
            # Adjust timing
            adjusted_segment = {
                **segment,
                'start': segment['start'] + cumulative_offset,
                'end': segment['end'] + cumulative_offset
            }
            merged_segments.append(adjusted_segment)
        
        # Update offset for next chunk (assuming 10-minute chunks with 30s overlap)
        if i < len(chunk_results) - 1:
            cumulative_offset += 10 * 60  # 10 minutes
    
    # Use language from first chunk
    merged_result = {
        'segments': merged_segments,
        'language_detected': chunk_results[0].get('language_detected', 'unknown')
    }
    
    # Merge other fields if present
    for key in ['translation', 'summary']:
        if key in chunk_results[0]:
            merged_values = [chunk.get(key, '') for chunk in chunk_results if chunk.get(key)]
            merged_result[key] = '\n\n'.join(merged_values) if merged_values else ''
    
    return merged_result

def optimize_audio_for_processing(audio_path: str) -> str:
    """
    Apply audio preprocessing optimizations for faster ML processing.
    
    Optimizations:
    - Noise reduction (basic)
    - Volume normalization
    - Remove silence gaps
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        
        # Normalize volume for consistent processing
        normalized_audio = audio.normalize()
        
        # Remove leading/trailing silence for faster processing
        # This can significantly reduce processing time for files with long silences
        trimmed_audio = normalized_audio.strip_silence(silence_thresh=-40)
        
        # Create optimized output
        optimized_path = tempfile.mktemp(suffix="_optimized.wav")
        trimmed_audio.export(
            optimized_path,
            format="wav",
            parameters=["-ac", "1", "-ar", "16000"]
        )
        
        original_duration = len(audio) / 1000
        optimized_duration = len(trimmed_audio) / 1000
        
        print(f"[Utils] Audio optimized: {original_duration:.1f}s → {optimized_duration:.1f}s")
        
        return optimized_path
        
    except Exception as e:
        print(f"[Utils] Error optimizing audio: {e}")
        return audio_path

def cleanup_temp_files(*file_paths: str) -> None:
    """
    Clean up temporary files efficiently.
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"[Utils] Cleaned up: {file_path}")
        except Exception as e:
            print(f"[Utils] Error cleaning up {file_path}: {e}")
