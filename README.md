---
title: Concisio App
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
---

# 🎵 Concisio App

A professional audio processing application that transcribes, diarizes, translates, and summarizes audio files using state-of-the-art AI models including WhisperX, PyAnnote, and OpenAI's GPT-4o.

## ✨ Core Features

### 🎤 **Advanced Audio Processing**
- **High-Quality Transcription**: Powered by WhisperX for superior accuracy
- **Speaker Diarization**: Identify and label different speakers automatically
- **Multi-format Support**: WAV, MP3, M4A, FLAC, OGG files
- **Automatic Language Detection**: Supports 14+ languages

### ⚡ **Performance Optimizations**
- **GPU Acceleration**: Automatic CUDA detection for 5-10x faster processing
- **Chunked Processing**: Handles large audio files (15+ minutes) efficiently
- **Optional Diarization**: Toggle speaker identification for faster processing
- **Fast Mode**: Reduced accuracy diarization for 30% speed improvement

### 🌍 **Translation & Summarization**
- **Multi-language Translation**: Translate to 14 supported languages
- **AI-Powered Summarization**: Custom prompts with GPT-4o
- **Prompt Enhancement**: AI-assisted prompt optimization
- **Professional Templates**: Pre-built summarization templates

### 🎨 **Modern Interface**
- **Streamlit-Powered**: Clean, responsive web interface
- **Real-time Progress**: Live updates during processing
- **Download Results**: Export transcriptions and summaries
- **Mobile-Friendly**: Works on desktop, tablet, and mobile

## 🏗️ Project Structure

```
concisio-app/
├── app.py                # Streamlit web interface
├── predict.py            # Core ML pipeline (Predictor class)
├── utils.py              # Audio processing utilities
├── requirements.txt      # Python dependencies
├── README.md             # This documentation
└── .env                  # Environment variables (create locally)
```

## 🚀 Quick Start

### **Option 1: Use on Hugging Face Spaces (Recommended)**
1. Visit our [Hugging Face Space](https://huggingface.co/spaces/your-username/concisio-app)
2. Upload your audio file
3. Configure settings (diarization, translation, summarization)
4. Click "Process Audio" and wait for results
5. Download your results

### **Option 2: Local Installation**

#### Prerequisites
- Python 3.8-3.11
- CUDA-compatible GPU (optional, for faster processing)
- OpenAI API key
- Hugging Face token (for diarization)

#### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/concisio-app
cd concisio-app
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_hugging_face_token_here
```

5. **Run the application**
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 🎛️ Usage Guide

### **Basic Workflow**
1. **Upload Audio**: Drag & drop or select your audio file
2. **Configure Settings**:
   - Toggle speaker diarization (on/off)
   - Choose diarization mode (Standard/Fast)
   - Select translation language (optional)
   - Customize summarization prompt (optional)
3. **Process**: Click "Process Audio" and monitor progress
4. **Review Results**: View transcription, translation, and summary
5. **Download**: Export results as text file

### **Performance Tips**
- **GPU Processing**: Use T4 small or higher for 5-10x speed improvement
- **Disable Diarization**: Skip speaker identification for 50% faster processing
- **Fast Mode**: Use fast diarization for 30% speed boost with slight accuracy trade-off
- **File Size**: Keep files under 15 minutes for optimal performance

### **Supported Languages**
Translation and transcription support for:
- English, Spanish, French, German, Italian
- Portuguese, Russian, Persian, Arabic, Greek
- Chinese, Japanese, Korean

## 🔧 Advanced Configuration

### **Hardware Requirements**

| Processing Mode | Hardware | 17MB File Time | Memory Usage |
|----------------|----------|----------------|--------------|
| CPU Basic | 2 vCPU, 16GB RAM | ~45-60 min | ~8GB |
| GPU T4 Small | 4 vCPU, 16GB VRAM | ~5-8 min | ~12GB |
| GPU T4 Medium | 8 vCPU, 30GB RAM | ~4-7 min | ~12GB |

### **Environment Variables**

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | For translation and summarization |
| `HF_TOKEN` | Yes | For speaker diarization models |

### **Model Information**
- **Transcription**: WhisperX (large-v2 on GPU, base on CPU)
- **Diarization**: PyAnnote.audio speaker diarization
- **Translation/Summary**: OpenAI GPT-4o
- **Languages**: Automatic detection + 14 target languages

## 📈 Performance Benchmarks

### **Processing Time Estimates**

| File Size | CPU (no diarization) | CPU (with diarization) | GPU T4 (with diarization) |
|-----------|---------------------|----------------------|--------------------------|
| 5MB (~3min) | ~2 minutes | ~5 minutes | ~1 minute |
| 17MB (~10min) | ~5 minutes | ~15 minutes | ~3 minutes |
| 50MB (~30min) | ~15 minutes | ~45 minutes | ~8 minutes |

## 🛠️ Deployment

### **Hugging Face Spaces**
1. Fork this repository
2. Create a new Streamlit Space on Hugging Face
3. Connect your GitHub repository
4. Add secrets in Space settings:
   - `OPENAI_API_KEY`
   - `HF_TOKEN`
5. Choose hardware (T4 small recommended)
6. Deploy automatically

### **Local Docker Deployment**
```bash
# Build image
docker build -t concisio-app .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e HF_TOKEN=your_token \
  concisio-app
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/your-username/concisio-app
cd concisio-app
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/your-username/concisio-app/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/concisio-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/concisio-app/discussions)

## 🙏 Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for advanced speech recognition
- [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [OpenAI](https://openai.com/) for GPT-4o translation and summarization
- [Streamlit](https://streamlit.io/) for the web interface

---

⭐ **Star this repository if you find it helpful!**
