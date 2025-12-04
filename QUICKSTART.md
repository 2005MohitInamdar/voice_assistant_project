# Quick Start Guide

## 5-Minute Setup

### 1. Get Your Murf API Key (2 minutes)

1. Go to https://murf.ai/
2. Sign up for a free account
3. Navigate to API settings
4. Copy your API key

### 2. Configure Environment (1 minute)

Create `backend/.env` file:
```
MURF_API_KEY=your_api_key_here
ASSEMBLYAI_API_KEY=
DEEPGRAM_API_KEY=
```

### 3. Install Dependencies (1 minute)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r backend/requirements.txt
```

### 4. Start the Application (1 minute)

**Terminal 1 - Backend:**
```bash
python backend/app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python -m http.server 3000
```

### 5. Open in Browser

Navigate to: http://localhost:3000

## First Test

1. Click "Start Recording"
2. Say: "Hello, what can you do?"
3. Wait for the response
4. Listen to the AI voice

## Troubleshooting

### "Backend not reachable"
- Check if backend is running on port 8000
- Look for errors in backend terminal

### "Microphone permission denied"
- Allow microphone access in browser
- Chrome works best for speech recognition

### "TTS Error"
- Verify API key in .env file
- Check internet connection
- Run `python backend/tts.py` to test API

### No audio playing
- Check browser audio settings
- Try clicking play button manually
- Check if audio file downloaded (look for file size)

## Tips for Demo

1. **Use Chrome** - Best speech recognition support
2. **Quiet environment** - Better speech detection
3. **Clear speech** - Speak clearly and at normal pace
4. **Wait for response** - System auto-stops after you finish
5. **Try different voices** - Show variety in demo
6. **Use advanced settings** - Demonstrate customization

## Common Commands to Try

- "Hello" - Greeting
- "What time is it?" - Time query
- "What's the date?" - Date query
- "What can you do?" - Capabilities
- "Tell me about Murf" - About the TTS
- "Thank you" - Polite response
- "Goodbye" - Farewell

## Keyboard Shortcuts

- **Space** - Start/Stop recording (when not in text field)

## Next Steps

1. Test all features (see TESTING_CHECKLIST.md)
2. Record demo video (see DEMO_SCRIPT.md)
3. Prepare LinkedIn post (see LINKEDIN_POST.md)
4. Submit before deadline!

## Support

If you encounter issues:
1. Check backend terminal for error messages
2. Check browser console (F12) for frontend errors
3. Verify API key is correct
4. Test TTS directly: `python backend/tts.py`

## Performance Tips

- First request may be slower (API warmup)
- Subsequent requests are faster
- Keep phrases under 100 words for best performance
- Close other tabs to free up resources
