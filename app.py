import subprocess
import sys
import os

# Install and configure cuDNN libraries
def setup_cudnn():
    # Check if already configured
    if os.environ.get('CUDNN_SETUP_COMPLETE'):
        return
        
    try:
        # Try to find where cudnn is installed
        import site
        site_packages = site.getsitepackages()[0]
        
        # Common paths where nvidia packages install libraries
        nvidia_paths = [
            os.path.join(site_packages, 'nvidia', 'cudnn', 'lib'),
            os.path.join(site_packages, 'nvidia', 'cudnn', 'lib64'),
            os.path.join(site_packages, 'nvidia', 'cublas', 'lib'),
            os.path.join(site_packages, 'nvidia', 'cuda_runtime', 'lib'),
        ]
        
        # Add all existing nvidia library paths
        ld_paths = []
        for path in nvidia_paths:
            if os.path.exists(path) and path not in ld_paths:
                ld_paths.append(path)
                print(f"Found NVIDIA library path: {path}")
        
        # Also check standard CUDA paths
        standard_paths = ['/usr/local/cuda/lib64', '/usr/lib/x86_64-linux-gnu']
        for path in standard_paths:
            if os.path.exists(path) and path not in ld_paths:
                ld_paths.append(path)
                print(f"Found standard CUDA path: {path}")
        
        if ld_paths:
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            # Only add paths that aren't already in LD_LIBRARY_PATH
            current_paths = current_ld.split(':') if current_ld else []
            new_paths = [p for p in ld_paths if p not in current_paths]
            
            if new_paths:
                new_ld = ':'.join(new_paths + current_paths) if current_paths else ':'.join(new_paths)
                os.environ['LD_LIBRARY_PATH'] = new_ld
                print(f"Updated LD_LIBRARY_PATH")
            
        # Mark as complete
        os.environ['CUDNN_SETUP_COMPLETE'] = '1'
            
    except Exception as e:
        print(f"Error setting up cuDNN: {e}")

# Only run setup once
if not os.environ.get('CUDNN_SETUP_COMPLETE'):
    # Your existing setup_cudnn function here
    setup_cudnn()

# Disable JIT only once
if not os.environ.get('PYTORCH_JIT_DISABLED'):
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['PYTORCH_JIT_DISABLED'] = '1'

# Debug: Check what paths were found
print("Checking for CUDA libraries...")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

def verify_cuda_setup():
    """Verify CUDA is properly set up"""
    try:
        import torch
        print("="*50)
        print("CUDA Setup Verification:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print("="*50)
    except Exception as e:
        print(f"Error verifying CUDA setup: {e}")

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import openai

# Call it right after imports
verify_cuda_setup()

# Load environment variables from .env file
load_dotenv()

# Add debugging
print("Starting Concisio App...")
print(f"Environment variables loaded: HF_TOKEN={'HF_TOKEN' in os.environ}, OPENAI_API_KEY={'OPENAI_API_KEY' in os.environ}")

# Initialize predictor with better error handling
@st.cache_resource
def load_predictor():
    try:
        from predict import Predictor
        predictor = Predictor()
        print("Predictor initialized successfully")
        return predictor
    except Exception as e:
        print(f"Error initializing Predictor: {e}")
        return None

# Initialize OpenAI client for prompt enhancement
@st.cache_resource
def get_openai_client():
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return openai.OpenAI(api_key=openai_key)
    return None

# Default summarization prompt
DEFAULT_SUMMARY_PROMPT = """Please provide a structured summary of the conversation with the following sections:
- **Background:** Briefly describe the context of the discussion.
- **Key Discussion Points:** List the main topics and what was said by each speaker.
- **Decisions Made:** Detail any decisions that were reached.
- **Next Steps:** Outline the action items, who is responsible, and the deadlines."""

# Dictionary of languages for the dropdown, with full names.
LANGUAGES = {
    "No Translation": "None",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Farsi (Persian)": "fa",
    "Arabic": "ar",
    "Greek": "el",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}

def enhance_prompt_with_ai(user_prompt, openai_client):
    """Enhance user's summarization prompt using GPT-4o"""
    if not openai_client:
        return "❌ OpenAI API not available for prompt enhancement."
    
    enhancement_prompt = """You are an expert prompt engineer specializing in text summarization. Your task is to take a user's basic summarization request and transform it into a professional, detailed prompt that will produce high-quality summaries.

Guidelines for enhancement:
1. Make the prompt more specific and actionable
2. Add structure and formatting requirements
3. Include instructions for handling different types of content
4. Specify the desired tone and style
5. Add requirements for key information extraction
6. Keep it professional but comprehensive

User's original prompt: "{user_prompt}"

Please enhance this into a professional summarization prompt that will work excellently with GPT-4o for transcribed audio content. Return only the enhanced prompt, no explanations."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert prompt engineer specializing in creating high-quality summarization prompts."},
                {"role": "user", "content": enhancement_prompt.format(user_prompt=user_prompt)}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error enhancing prompt: {str(e)}"

def estimate_processing_time(file_size_mb, include_diarization=True, gpu_available=False, fast_diarization=False):
    """Estimate processing time based on file size and hardware"""
    if gpu_available:
        # GPU processing times
        base_time = file_size_mb * 0.05  # ~0.05 minutes per MB with GPU
        if include_diarization:
            if fast_diarization:
                base_time *= 1.8  # Fast diarization
            else:
                base_time *= 2.5  # Standard diarization
    else:
        # CPU processing times (current)
        base_time = file_size_mb * 0.3  # ~0.3 minutes per MB for transcription
        if include_diarization:
            if fast_diarization:
                base_time *= 2.0  # Fast diarization
            else:
                base_time *= 2.5  # Standard diarization
    
    return max(1, int(base_time))

def process_audio_file(audio_file, target_language, custom_prompt, predictor, include_diarization=True, fast_diarization=False, progress_callback=None):
    """
    Process the uploaded audio file for transcription, translation, and summarization.
    """
    # Check if predictor is available
    if predictor is None:
        return "Error: Predictor not initialized. Please check the logs for initialization errors.", "", "", ""
    
    # Debug: Initial state
    print(f"\n{'='*50}")
    print(f"Processing started at: {datetime.now()}")
    print(f"Audio file: {audio_file.name}")
    print(f"Target language: {target_language}")
    print(f"Include diarization: {include_diarization}")
    print(f"Custom prompt provided: {bool(custom_prompt and custom_prompt.strip())}")
    
    if audio_file is None:
        print("ERROR: No audio file provided")
        return "Please upload an audio file first.", "", "", ""
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{audio_file.name}"
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # Debug: File info
    if os.path.exists(temp_path):
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        print(f"File type: {os.path.splitext(temp_path)[1]}")
    
    try:
        print("\nCalling predictor.predict()...")
        
        result = predictor.predict(
            audio_file_path=temp_path,
            target_language=target_language,
            custom_summary_prompt=custom_prompt,
            include_diarization=include_diarization,
            fast_diarization=fast_diarization,
            progress_callback=progress_callback
        )
        
        print(f"Predictor returned successfully")
        print(f"Result keys: {list(result.keys())}")
        
        if "error" in result:
            error_message = f"An error occurred: {result['error']}"
            print(f"ERROR in result: {result['error']}")
            return error_message, "", "", ""
        
        # Debug: Process results
        segments = result.get("segments", [])
        print(f"\nNumber of segments: {len(segments)}")
        if segments:
            print(f"First segment: {segments[0]}")
            
        if include_diarization:
            segments_formatted = "\n".join(
                [f"[{s['start']:.2f}-{s['end']:.2f}] {s['speaker']}: {s['text'].strip()}" 
                 for s in segments]
            )
        else:
            segments_formatted = "\n".join(
                [f"[{s['start']:.2f}-{s['end']:.2f}] {s['text'].strip()}" 
                 for s in segments]
            )
        
        language_detected = result.get("language_detected", "N/A")
        print(f"Language detected: {language_detected}")
        
        translation = result.get("translation", "Not requested or an error occurred.")
        summary = result.get("summary", "Not requested or an error occurred.")
        
        print(f"Translation provided: {bool(translation and translation != 'Not requested or an error occurred.')}")
        print(f"Summary provided: {bool(summary and summary != 'Not requested or an error occurred.')}")
        
        print(f"Processing completed at: {datetime.now()}")
        print(f"{'='*50}\n")
        
        return segments_formatted, language_detected, translation, summary
        
    except Exception as e:
        print(f"\nEXCEPTION in process_audio_file: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", "", ""
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Concisio App",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Concisio App")
    st.markdown("Upload an audio file to transcribe, diarize, and optionally translate or summarize.")
    
    # Load predictor and OpenAI client with error handling
    try:
        predictor = load_predictor()
        openai_client = get_openai_client()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("This might be a GPU memory issue. Try refreshing the page or contact support.")
        return
    
    if predictor is None:
        st.error("⚠️ Predictor not initialized. Please check the logs for initialization errors.")
        st.info("**Troubleshooting tips:**")
        st.info("- Make sure your Space has sufficient GPU memory")
        st.info("- Try refreshing the page")
        st.info("- Check if all required environment variables are set")
        return
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # Audio upload
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
        )
        
        # Show file info and estimated processing time
        if audio_file is not None:
            file_size_mb = len(audio_file.getbuffer()) / (1024 * 1024)
            st.info(f"📁 File size: {file_size_mb:.1f} MB")
        
        # Performance settings
        st.subheader("⚡ Performance Settings")
        
        # Diarization option
        include_diarization = st.checkbox(
            "🎭 Enable Speaker Diarization", 
            value=True,
            help="Identify different speakers in the audio"
        )
        
        if include_diarization:
            # Diarization speed option
            diarization_mode = st.radio(
                "Diarization Mode:",
                ["Standard (More Accurate)", "Fast (Less Accurate)"],
                index=0,
                help="Fast mode reduces processing time but may be less accurate"
            )
            
            if diarization_mode == "Fast (Less Accurate)":
                st.info("🚀 Fast mode enabled - processing time reduced by ~30%")
            else:
                st.warning("⚠️ Standard diarization can significantly increase processing time")
        
        # GPU status
        gpu_available = predictor.device == "cuda" if predictor else False
        if gpu_available:
            st.success("🚀 GPU acceleration enabled")
        else:
            st.warning("⚠️ Running on CPU - consider upgrading to GPU Space for 5-10x faster processing")
            if st.button("📖 How to enable GPU?"):
                st.info("""
                **To enable GPU acceleration:**
                1. Go to your Space settings
                2. Change Hardware from 'CPU basic' to 'GPU T4 small' or higher
                3. Restart your Space
                
                **Performance improvement with GPU:**
                - 17MB file: ~45min → ~5-8min
                - Diarization: ~2-3x faster
                """)
        
        # Show estimated processing time
        if audio_file is not None:
            fast_diarization = include_diarization and diarization_mode == "Fast (Less Accurate)"
            estimated_time = estimate_processing_time(file_size_mb, include_diarization, gpu_available, fast_diarization)
            st.info(f"⏱️ Estimated processing time: ~{estimated_time} minutes")
        
        # Language selection
        target_language = st.selectbox(
            "Translate to (Optional)",
            options=list(LANGUAGES.keys()),
            index=0
        )
        
        st.divider()
        
        # Summarization section
        st.subheader("📋 Summarization")
        
        # Default prompt toggle
        use_default_prompt = st.checkbox(
            "Use default summarization prompt",
            value=True,
            help="Use our professionally crafted default prompt"
        )
        
        # Summary prompt
        if use_default_prompt:
            custom_prompt = st.text_area(
                "Summarization Prompt",
                value=DEFAULT_SUMMARY_PROMPT,
                height=200,
                help="This is our default prompt. You can modify it or uncheck 'Use default' to start fresh."
            )
        else:
            custom_prompt = st.text_area(
                "Summarization Prompt (Leave blank to skip)",
                value="",
                height=200,
                placeholder="Enter your custom summarization instructions here..."
            )
        
        # AI Enhancement button
        if custom_prompt and custom_prompt.strip():
            if st.button("✨ Enhance Prompt with AI", help="Use GPT-4o to make your prompt more professional and effective"):
                if openai_client:
                    with st.spinner("🤖 Enhancing your prompt..."):
                        enhanced = enhance_prompt_with_ai(custom_prompt, openai_client)
                        if not enhanced.startswith("❌"):
                            st.session_state['enhanced_prompt'] = enhanced
                            st.success("✅ Prompt enhanced! Check the text area below.")
                        else:
                            st.error(enhanced)
                else:
                    st.error("❌ OpenAI API not available for prompt enhancement.")
        
        # Show enhanced prompt if available
        if 'enhanced_prompt' in st.session_state:
            st.subheader("🚀 Enhanced Prompt")
            enhanced_display = st.text_area(
                "Enhanced Prompt (copy to use above)",
                value=st.session_state['enhanced_prompt'],
                height=150,
                help="Copy this enhanced prompt to the main prompt area above"
            )
            if st.button("📋 Use Enhanced Prompt"):
                st.session_state['custom_prompt_value'] = st.session_state['enhanced_prompt']
                st.rerun()
        
        st.divider()
        
        # Process button
        process_button = st.button(
            "🚀 Process Audio",
            type="primary",
            use_container_width=True,
            disabled=(audio_file is None)
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Transcription & Diarization")
        transcription_placeholder = st.empty()
        
        st.subheader("🌍 Detected Language")
        language_placeholder = st.empty()
    
    with col2:
        st.subheader("🔄 Translation")
        translation_placeholder = st.empty()
        
        st.subheader("📋 Summary")
        summary_placeholder = st.empty()
    
    # Handle enhanced prompt usage
    if 'custom_prompt_value' in st.session_state:
        custom_prompt = st.session_state['custom_prompt_value']
        del st.session_state['custom_prompt_value']
    
    # Process audio when button is clicked
    if process_button and audio_file is not None:
        # Get diarization mode setting
        fast_mode = include_diarization and diarization_mode == "Fast (Less Accurate)"
        estimated_time = estimate_processing_time(
            len(audio_file.getbuffer()) / (1024 * 1024), 
            include_diarization, 
            gpu_available, 
            fast_mode
        )
        
        with st.spinner(f"Processing audio... This may take up to {estimated_time} minutes."):
            # Show progress info
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Create a progress callback function
                def update_progress(step, progress_pct):
                    status_text.text(step)
                    progress_bar.progress(progress_pct)
                
                status_text.text("🎵 Loading audio file...")
                progress_bar.progress(10)
                
                language_code = LANGUAGES.get(target_language, "None")
                
                status_text.text("🎤 Starting transcription...")
                progress_bar.progress(30)
                
                if include_diarization:
                    status_text.text("🎭 Processing speaker diarization...")
                    progress_bar.progress(60)
                else:
                    progress_bar.progress(70)
                
                if custom_prompt and custom_prompt.strip():
                    status_text.text("📋 Generating summary...")
                    progress_bar.progress(85)
                
                # Process the audio
                transcription, language_detected, translation, summary = process_audio_file(
                    audio_file, language_code, custom_prompt, predictor, include_diarization, fast_mode,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Clear progress after a moment
                import time
                time.sleep(1)
                progress_container.empty()
            
            # Display results
            with transcription_placeholder.container():
                st.text_area(
                    "Transcription Result",
                    value=transcription,
                    height=300,
                    label_visibility="collapsed"
                )
            
            with language_placeholder.container():
                st.info(f"Detected: {language_detected}")
            
            with translation_placeholder.container():
                st.text_area(
                    "Translation Result",
                    value=translation,
                    height=150,
                    label_visibility="collapsed"
                )
            
            with summary_placeholder.container():
                st.text_area(
                    "Summary Result",
                    value=summary,
                    height=150,
                    label_visibility="collapsed"
                )
            
            # Show download button for results
            results_text = f"""TRANSCRIPTION & DIARIZATION:
{transcription}

DETECTED LANGUAGE: {language_detected}

TRANSLATION:
{translation}

SUMMARY:
{summary}
"""
            
            st.download_button(
                label="💾 Download Results",
                data=results_text,
                file_name=f"concisio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("✅ Processing completed!")

if __name__ == "__main__":
    main()
