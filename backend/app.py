from flask import Flask, request, send_file, jsonify, make_response
import io
import os
from dotenv import load_dotenv
from tts import murf_synthesize, get_available_voices
from logic import generate_reply
from asr import get_asr_status, get_preferred_asr
from flask_cors import CORS
from memory import add_to_memory, clear_memory
import base64

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

app = Flask(__name__)

CORS(app)

@app.route("/")
def home():
    return "Voice Agent Backend Running"

@app.route("/config", methods=["GET"])
def config():
    # Tell frontend what ASR options are available
    # This way frontend can adapt based on what's configured
    asr_status = get_asr_status()
    preferred = get_preferred_asr()
    return jsonify({
        "asr": asr_status,
        "preferred_asr": preferred,
        "deepgram_key": os.getenv("DEEPGRAM_API_KEY") if asr_status["deepgram"] else None
    })

@app.route("/voices", methods=["GET"])
def voices():

    voices = get_available_voices()
    return jsonify(voices)

@app.route("/synthesize", methods=["POST"])
def synthesize():
    # Main endpoint - takes user speech text and returns audio response
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return "Missing 'text' field", 400

        user_text = data["text"]
        
        voice_id = data.get("voiceId", "en-US-ken") 
        style = data.get("style", "Conversational")
        rate = data.get("rate", 0)  # -50 to 50, controls speed
        pitch = data.get("pitch", 0)  # -50 to 50, controls tone
        
        print(f"\n📝 Request: '{user_text[:50]}...'")
        print(f"   Voice: {voice_id}, Style: {style}, Rate: {rate}, Pitch: {pitch}")

        # First figure out what to say back
        reply_text = generate_reply(user_text)
        print(f"💬 Reply: '{reply_text[:50]}...'")

        
        reply_text = generate_reply(user_text)
        print(f"💬 Reply: '{reply_text[:50]}...'")
        
        # NEW: Add this conversation turn to memory
        add_to_memory(user_text, reply_text)
        
        
        
        # Then convert that text to actual audio using Murf API
        audio_bytes, mime = murf_synthesize(
            reply_text, 
            voice_id=voice_id,
            style=style,
            rate=rate,
            pitch=pitch
        )

        if audio_bytes is None:
            print("❌ TTS failed - returning error")
            return "TTS Error: Could not generate audio", 500

        print(f"✅ Sending {len(audio_bytes):,} bytes of audio\n")

        if audio_bytes is None:
            print("TTS failed - returning error")
            return "TTS Error: Could not generate audio", 500

        print(f"Sending {len(audio_bytes):,} bytes of audio + reply text\n")

        # Convert audio to base64 so we can send it in JSON
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Return JSON with both reply text and audio
        return jsonify({
            "reply_text": reply_text.strip(),
            "audio_base64": audio_base64,
            "mime_type": mime
        })
    except Exception as e:
        print(f"❌ Error in synthesize endpoint: {e}")
        return f"Server error: {str(e)}", 500


@app.route("/clear_memory", methods=["POST"])
def clear_memory_endpoint():
    clear_memory()
    return jsonify({"status": "Memory cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

