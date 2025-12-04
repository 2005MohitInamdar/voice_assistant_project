import os
import requests
from dotenv import load_dotenv
import json

load_dotenv('.env')

MURF_API_KEY = os.getenv("MURF_API_KEY")
print(f"Fetching voices with API key: {MURF_API_KEY[:20]}...\n")

headers = {
    "api-key": MURF_API_KEY,
    "Accept": "application/json"
}

try:
    response = requests.get(
        "https://api.murf.ai/v1/speech/voices",
        headers=headers,
        timeout=10
    )
    
    if response.status_code == 200:
        voices = response.json()
        print(f"✅ Found {len(voices)} voices\n")
        
        # Filter for English female voices
        print("FEMALE VOICES (English):")
        print("-" * 60)
        for voice in voices:
            if isinstance(voice, dict):
                voice_id = voice.get('voiceId', 'N/A')
                name = voice.get('name', 'N/A')
                gender = voice.get('gender', 'N/A')
                language = voice.get('language', 'N/A')
                
                if 'female' in gender.lower() and 'en-' in voice_id.lower():
                    print(f"ID: {voice_id:25} Name: {name:15} Lang: {language}")
        
        print("\n" + "=" * 60)
        print("MALE VOICES (English):")
        print("-" * 60)
        for voice in voices:
            if isinstance(voice, dict):
                voice_id = voice.get('voiceId', 'N/A')
                name = voice.get('name', 'N/A')
                gender = voice.get('gender', 'N/A')
                language = voice.get('language', 'N/A')
                
                if 'male' in gender.lower() and 'female' not in gender.lower() and 'en-' in voice_id.lower():
                    print(f"ID: {voice_id:25} Name: {name:15} Lang: {language}")
                    
    else:
        print(f"❌ Failed: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
