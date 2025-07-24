import os
import tempfile
from pydub import AudioSegment
import pandas as pd

def convert_to_wav(input_path):
    """
    Converts any audio file supported by ffmpeg to a 16kHz mono WAV file.
    
    Args:
        input_path (str): Path to the input audio file.

    Returns:
        str: Path to the converted temporary WAV file.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_path)
    
    # Set to mono
    audio = audio.set_channels(1)
    
    # Set to 16kHz sample rate
    audio = audio.set_frame_rate(16000)
    
    # Create a temporary file to store the WAV output
    # The file will be created in the default temporary directory
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    # Export the audio to WAV format
    audio.export(temp_wav.name, format="wav")
    
    return temp_wav.name

def split_chunks(wav_path, max_duration_seconds=1200):
    """
    Splits a WAV file into smaller chunks.
    Note: This is not currently used in predict.py but is included for future use
    with very long audio files, as per the original specification.

    Args:
        wav_path (str): Path to the input WAV file.
        max_duration_seconds (int): Maximum duration of each chunk in seconds (default 20 mins).

    Returns:
        list: A list of paths to the audio chunks.
    """
    audio = AudioSegment.from_wav(wav_path)
    duration_ms = len(audio)
    max_duration_ms = max_duration_seconds * 1000
    
    chunks = []
    for i in range(0, duration_ms, max_duration_ms):
        chunk = audio[i:i + max_duration_ms]
        chunk_path = f"temp_chunk_{i // max_duration_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        
    return chunks

def merge_text_diarization(aligned_segments, diarize_segments):
    """
    Merges aligned transcription segments with diarization results to assign speakers.

    Args:
        aligned_segments (list of dicts): The result from whisperx.align.
        diarize_segments (pd.DataFrame): The result from whisperx.DiarizationPipeline.

    Returns:
        list: A list of segments, each with 'text', 'start', 'end', and 'speaker'.
    """
    from whisperx.lib import assign_word_speakers # Import locally to avoid circular deps if utils is larger

    # Assign speaker labels to each word
    result_segments_w_speakers = assign_word_speakers(diarize_segments, aligned_segments)

    # Re-segment the text based on speaker turns
    final_segments = []
    current_speaker = None
    segment_text = ""
    segment_start = 0

    for segment in result_segments_w_speakers["segments"]:
        if "words" not in segment:
            continue

        for word_info in segment["words"]:
            if "speaker" not in word_info:
                # If a word has no speaker, continue with the previous one
                word_info["speaker"] = current_speaker if current_speaker else "UNKNOWN"

            if current_speaker is None:
                # Start of the first segment
                current_speaker = word_info["speaker"]
                segment_start = word_info["start"]

            if current_speaker != word_info["speaker"]:
                # End of the current segment, start of a new one
                final_segments.append({
                    "text": segment_text.strip(),
                    "start": segment_start,
                    "end": word_info["start"], # The new word's start is the previous segment's end
                    "speaker": current_speaker
                })
                # Start the new segment
                current_speaker = word_info["speaker"]
                segment_text = ""
                segment_start = word_info["start"]

            segment_text += word_info["word"] + " "
    
    # Add the last segment
    if segment_text:
        # To get the end time of the last segment, we find the last word's end time
        last_word_end = result_segments_w_speakers["segments"][-1]["words"][-1]["end"]
        final_segments.append({
            "text": segment_text.strip(),
            "start": segment_start,
            "end": last_word_end,
            "speaker": current_speaker
        })

    return final_segments


def build_translate_prompt(text, target_language):
    """
    Builds the prompt for the OpenAI API to perform translation.

    Args:
        text (str): The text to be translated.
        target_language (str): The target language code (e.g., 'es', 'fr').

    Returns:
        str: The formatted prompt.
    """
    return (
        f"Translate the following text into {target_language}. "
        "Preserve the original meaning and intent. "
        "Do not add any extra commentary or explanation, only provide the translation.\n\n"
        f"Text to translate:\n---\n{text}"
    )

def build_summary_prompt(text, custom_prompt):
    """
    Builds the prompt for the OpenAI API to perform summarization.

    Args:
        text (str): The text to be summarized.
        custom_prompt (str): The user-provided prompt for summarization.

    Returns:
        str: The formatted prompt.
    """
    return (
        f"You are a helpful assistant that summarizes text based on a user's request.\n\n"
        f"User's request: '{custom_prompt}'\n\n"
        f"Based on this request, please summarize the following text:\n---\n{text}"
    )
