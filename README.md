# AI Voice Agent - Murf Falcon TTS Integration

A real-time conversational voice agent built for Techfest 2025-26 Murf Voice Agent Hackathon. This application demonstrates the power of Murf Falcon API for fast, natural speech generation in an interactive voice-driven experience.

## Overview

This voice agent allows users to have natural conversations through speech. Users speak into their microphone, and the agent responds with synthesized speech using Murf Falcon's industry-leading TTS technology. The application features customizable voice settings including multiple voices, speaking styles, and audio adjustments.

## Features

- **Real-time Speech Recognition**: Uses browser-based Web Speech API for instant voice input
- **Murf Falcon TTS Integration**: Generates natural, production-grade speech responses
- **Multiple Voice Options**: 8 verified working voices across US and UK accents
  - Wayne, Cooper, Terrell, Hugo (Male)
  - Natalie, Samantha, Claire, Ruby (Female)
- **AI-Powered Responses**: Uses Groq (free) or OpenAI for intelligent conversation
- **Customizable Speech Settings**:
  - Voice selection (male/female, different accents)
  - Speaking styles (Conversational, Professional, Friendly, etc.)
  - Speech rate control (-50 to +50)
  - Pitch adjustment (-50 to +50)
- **Modern UI**: Built with React and Tailwind CSS with smooth scrolling
- **Secure API Management**: Environment variables for API key security
- **Fixed Issues**: All voice synthesis errors resolved, proper error handling implemented

## Tech Stack

**Frontend:**
- React 18
- Tailwind CSS
- Web Speech API (for ASR)

**Backend:**
- Python 3.x
- Flask (REST API)
- Murf Falcon API (TTS)
- Groq API / OpenAI (LLM for responses)
- LangChain (LLM integration)
- python-dotenv (environment management)
- Flask-CORS (cross-origin support)

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Modern web browser (Chrome recommended for best speech recognition)
- Murf AI account with API key
- Groq API key (free) OR OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/2005MohitInamdar/voice_assistant_project
cd voice_assistant_project
```

2. **Set up Python virtual environment**

```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the `backend` folder:
```
MURF_API_KEY=
# GROQ_API_KEY=
HF_TOKEN=
GROQ_API_KEY=
```

**Get your API keys:**
- Murf API key: https://murf.ai/
- Groq API key (FREE): https://console.groq.com
- HF_TOKEN (Huggingface token) (FREE): https://huggingface.co/settings/tokens 
**Note:** Groq is recommended as it's free and fast. Alternatively, you can use OpenAI by setting `OPENAI_API_KEY` instead.

5. **Start the backend server**
```bash
python app.py
```

The backend will run on `http://localhost:8000`

6. **Install frontend dependencies**

In a new terminal:
```bash
cd frontend
npm install
npm install bootstrap
npm install tailwindcss @tailwindcss/vite
```

7. **Start the frontend server**
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173/`

8. **Open in browser**

Navigate to `http://localhost:5173` in Chrome or another modern browser.

## Usage

1. Click "Start Recording" to begin speaking
2. Speak your message clearly
3. Click "Stop" when finished
4. The agent will process your speech and respond with audio
5. Customize voice settings using the dropdowns and sliders

## API Integration

### Murf Falcon API

This project uses the Murf Falcon API for text-to-speech generation. Key features:

- **Endpoint**: `https://api.murf.ai/v1/speech/generate`
- **Authentication**: API key via header
- **Response Time**: Real-time generation (typically < 2 seconds)
- **Quality**: Production-grade, natural-sounding speech
- **Free Tier**: 1,000,000 characters for new accounts

### API Request Structure

```python
payload = {
    "voiceId": "en-US-ken",
    "style": "Conversational",
    "text": "Your text here",
    "rate": 0,
    "pitch": 0,
    "sampleRate": 48000,
    "format": "MP3",
    "channelType": "STEREO"
}
```

## Project Structure

```
voice_agent/
├── backend/
│   ├── app.py              # Flask API server
│   ├── tts.py              # Murf TTS integration
│   ├── asr.py              # ASR configuration
│   ├── logic.py            # Conversation logic
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables (not in repo)
├── frontend/
│   ├── index.html         # Main HTML
│   └── app.jsx            # React application
└── README.md              # This file
```

## Security

- API keys are stored in `.env` file (not committed to repository)
- Environment variables loaded using `python-dotenv`
- CORS configured for local development only
- No sensitive data exposed to frontend


## LinkedIn Post

[Link to LinkedIn post will be added here]

## Built With Murf Falcon

This project is built using **Murf Falcon – the consistently fastest TTS API**.

Falcon delivers:
- Real-time speech generation
- Natural, human-like voices
- Production-grade quality
- Reliable performance at scale

## Future Enhancements

- Integration with Deepgram/AssemblyAI for improved ASR
- Multi-language support
- Conversation history
- Custom voice training
- Integration with LLMs for smarter responses
- Mobile app version

## Team

- Gaurang Udgirkar
- Mohit Inamdar

## Acknowledgments

- Built for Techfest 2025-26 Murf Voice Agent Hackathon
- Powered by Murf Falcon API
- Thanks to @Murf AI for providing the TTS technology

## License

MIT License

## Tags

`murf-ai` `voice-agent` `tts` `speech-synthesis` `hackathon` `techfest-2025`







