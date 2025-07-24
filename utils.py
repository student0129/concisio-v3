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
    print(f"[Utils] Starting WAV conversion for: {input_path}")
    # Load the audio file
    audio = AudioSegment.from_file(input_path)
    print(f"[Utils]  > Original audio info: {audio.channels} channels, {audio.frame_rate} Hz")
    
    # Set to mono
    audio = audio.set_channels(1)
    
    # Set to 16kHz sample rate
    audio = audio.set_frame_rate(16000)
    print(f"[Utils]  > Converted audio info: {audio.channels} channel(s), {audio.frame_rate} Hz")
    
    # Create a temporary file to store the WAV output
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    # Export the audio to WAV format
    audio.export(temp_wav.name, format="wav")
    print(f"[Utils]  > Exported to temporary WAV file: {temp_wav.name}")
    
    return temp_wav.name

def merge_text_diarization(aligned_segments, diarize_segments):
    """
    Merges aligned transcription segments with diarization results to assign speakers.

    Args:
        aligned_segments (list of dicts): The result from whisperx.align.
        diarize_segments (pd.DataFrame): The result from whisperx.DiarizationPipeline.

    Returns:
        list: A list of segments, each with 'text', 'start', 'end', and 'speaker'.
    """
    print("[Utils] Starting merge of transcription and diarization...")
    print(f"[Utils]  > Received {len(aligned_segments)} aligned segments.")
    print(f"[Utils]  > Received {len(diarize_segments)} diarization segments.")

    from whisperx.lib import assign_word_speakers

    # Assign speaker labels to each word
    result_segments_w_speakers = assign_word_speakers(diarize_segments, aligned_segments)
    print("[Utils]  > Assigned speakers to words.")
    if result_segments_w_speakers["segments"] and "words" in result_segments_w_speakers["segments"][0]:
         print(f"[Utils]  > Sample of first segment with word speakers: {result_segments_w_speakers['segments'][0]['words'][:5]}")


    # Re-segment the text based on speaker turns
    final_segments = []
    current_speaker = None
    segment_text = ""
    segment_start = 0

    if not result_segments_w_speakers.get("segments"):
        print("[Utils]  > No segments found after speaker assignment. Returning empty list.")
        return final_segments

    # Find the first word with a speaker to initialize
    for segment in result_segments_w_speakers.get("segments", []):
        if "words" in segment:
            for word_info in segment.get("words", []):
                if "speaker" in word_info and "start" in word_info:
                    current_speaker = word_info["speaker"]
                    segment_start = word_info["start"]
                    break
            if current_speaker:
                break
    
    print(f"[Utils]  > Starting re-segmentation with initial speaker: {current_speaker}")

    for segment in result_segments_w_speakers["segments"]:
        if "words" not in segment:
            continue

        for word_info in segment["words"]:
            if "speaker" not in word_info or "start" not in word_info or "word" not in word_info:
                continue

            if current_speaker != word_info["speaker"]:
                final_segments.append({
                    "text": segment_text.strip(),
                    "start": segment_start,
                    "end": word_info["start"],
                    "speaker": current_speaker
                })
                current_speaker = word_info["speaker"]
                segment_text = ""
                segment_start = word_info["start"]

            segment_text += word_info["word"] + " "
    
    # Add the last segment
    if segment_text:
        last_word_end = result_segments_w_speakers["segments"][-1]["words"][-1].get("end", segment_start)
        final_segments.append({
            "text": segment_text.strip(),
            "start": segment_start,
            "end": last_word_end,
            "speaker": current_speaker
        })
    
    print(f"[Utils] Merge complete. Generated {len(final_segments)} final segments.")
    return final_segments


def build_translate_prompt(text, target_language):
    """
    Builds the prompt for the OpenAI API to perform translation.
    """
    prompt = (
        f"Translate the following text into {target_language}. "
        "Preserve the original meaning and intent. "
        "Do not add any extra commentary or explanation, only provide the translation.\n\n"
        f"Text to translate:\n---\n{text}"
    )
    print(f"[Utils] Built translation prompt (first 100 chars): '{prompt[:100]}...'")
    return prompt

def build_summary_prompt(text, custom_prompt):
    """
    Builds the prompt for the OpenAI API to perform summarization.
    """
    prompt = (
        f"You are a helpful assistant that summarizes text based on a user's request.\n\n"
        f"User's request: '{custom_prompt}'\n\n"
        f"Based on this request, please summarize the following text:\n---\n{text}"
    )
    print(f"[Utils] Built summary prompt (first 100 chars): '{prompt[:100]}...'")
    return prompt
