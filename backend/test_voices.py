import os
import requests
from dotenv import load_dotenv

load_dotenv('.env')

MURF_API_KEY = os.getenv("MURF_API_KEY")
print(f"Testing with API key: {MURF_API_KEY[:20]}...")

# Test different female voice IDs
female_voices_to_test = [
    "en-US-natalie",
    "en-US-lily", 
    "en-US-ruby",
    "en-US-sarah",
    "en-US-lisa",
    "en-US-emma",
    "en-US-kate",
    "en-US-olivia",
    "en-US-amelia"
]

headers = {
    "api-key": MURF_API_KEY,
    "Content-Type": "application/json"
}

test_text = "Hello, this is a test."

for voice_id in female_voices_to_test:
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
    
    print(f"\nTesting voice: {voice_id}")
    try:
        response = requests.post(
            "https://api.murf.ai/v1/speech/generate",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✅ SUCCESS - {voice_id} works!")
        else:
            print(f"  ❌ FAILED - Status {response.status_code}")
            print(f"     Error: {response.text[:100]}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
