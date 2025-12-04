import os
import requests
from dotenv import load_dotenv

load_dotenv('.env')

MURF_API_KEY = os.getenv("MURF_API_KEY")
print(f"Testing female voices with API key: {MURF_API_KEY[:20]}...\n")

# Test the female voices we're using
female_voices = [
    "en-US-natalie",
    "en-US-samantha",
    "en-US-claire",
    "en-UK-ruby"
]

headers = {
    "api-key": MURF_API_KEY,
    "Content-Type": "application/json"
}

test_text = "Hello, this is a test of my voice."

for voice_id in female_voices:
    payload = {
        "voiceId": voice_id,
        "style": "Conversational",
        "text": test_text,
        "rate": 0,
        "pitch": 0,
        "sampleRate": 48000,
        "format": "MP3",
        "channelType": "STEREO"
    }
    
    print(f"Testing: {voice_id:20}", end=" ")
    try:
        response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ SUCCESS")
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(f"   Error: {response.text[:150]}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

print("\n✅ All female voices tested!")
