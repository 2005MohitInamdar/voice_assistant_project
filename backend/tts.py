import os
import requests
from dotenv import load_dotenv
import time

# Get the .env file from same folder as this script
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"🔍 Loading .env from: {env_path}")
load_dotenv(env_path)

MURF_API_KEY = os.getenv("MURF_API_KEY")
print(f"🔑 API Key loaded: {'Yes' if MURF_API_KEY else 'No'}")
if MURF_API_KEY:
    print(f"🔑 API Key starts with: {MURF_API_KEY[:10]}...")

# Murf's API endpoint for generating speech
MURF_TTS_URL = "https://api.murf.ai/v1/speech/generate"

# Defaults that work well for most cases
# Note: Murf voice IDs might be different - using a safe default
DEFAULT_VOICE_ID = "en-US-ken"
DEFAULT_FORMAT = "MP3" 
DEFAULT_SAMPLE_RATE = 48000  
REQUEST_TIMEOUT = 30  
MAX_RETRIES = 3  


def get_available_voices():
    # List of actual Murf voices - verified from API
    # These are real voice IDs that work with your Murf API key
    return [
        {
            "id": "en-US-wayne", 
            "name": "Wayne", 
            "gender": "Male", 
            "language": "English (US)", 
            "style": "Professional",
            "murf_voice_id": "en-US-wayne",  
            "accent": "American"
        },
        {
            "id": "en-US-natalie", 
            "name": "Natalie", 
            "gender": "Female", 
            "language": "English (US)", 
            "style": "Friendly",
            "murf_voice_id": "en-US-natalie",
            "accent": "American"
        },
        {
            "id": "en-US-cooper", 
            "name": "Cooper", 
            "gender": "Male", 
            "language": "English (US)", 
            "style": "Casual",
            "murf_voice_id": "en-US-cooper",
            "accent": "American"
        },
        {
            "id": "en-US-samantha", 
            "name": "Samantha", 
            "gender": "Female", 
            "language": "English (US)", 
            "style": "Warm",
            "murf_voice_id": "en-US-samantha",
            "accent": "American"
        },
        {
            "id": "en-US-terrell", 
            "name": "Terrell", 
            "gender": "Male", 
            "language": "English (US)", 
            "style": "Confident",
            "murf_voice_id": "en-US-terrell",
            "accent": "American"
        },
        {
            "id": "en-US-claire", 
            "name": "Claire", 
            "gender": "Female", 
            "language": "English (US)", 
            "style": "Energetic",
            "murf_voice_id": "en-US-claire",
            "accent": "American"
        },
        {
            "id": "en-UK-hugo", 
            "name": "Hugo", 
            "gender": "Male", 
            "language": "English (UK)", 
            "style": "Formal",
            "murf_voice_id": "en-UK-hugo",
            "accent": "British"
        },
        {
            "id": "en-UK-ruby", 
            "name": "Ruby", 
            "gender": "Female", 
            "language": "English (UK)", 
            "style": "Elegant",
            "murf_voice_id": "en-UK-ruby",
            "accent": "British"
        },
    ]

def murf_synthesize(text: str, voice_id: str = "en-US-ken", style: str = "Conversational", rate: int = 0, pitch: int = 0):

    available_voices = get_available_voices()
    
    # Find the voice in our list
    murf_voice_id = "en-US-ken"  # Default fallback
    for voice in available_voices:
        if voice["id"] == voice_id:
            murf_voice_id = voice["murf_voice_id"]
            break
    
    # Basic validation - can't do anything without API key
    if not MURF_API_KEY:
        print("❌ ERROR: Missing MURF_API_KEY in .env file")
        print("💡 Add: MURF_API_KEY=your_api_key_here to your .env file")
        return None, None
    
    # Also can't synthesize nothing
    if not text or not text.strip():
        print("⚠️ WARNING: Empty text provided for synthesis")
        return None, None
    
    text = text.strip()
    
    # Murf has limits on text length, so truncate if needed
    MAX_TEXT_LENGTH = 5000
    if len(text) > MAX_TEXT_LENGTH:
        print(f"⚠️ WARNING: Text too long ({len(text)} chars). Truncating to {MAX_TEXT_LENGTH} chars.")
        text = text[:MAX_TEXT_LENGTH]
    
    print(f"🎤 Synthesizing with voice '{voice_id}' -> Murf ID: '{murf_voice_id}'")
    print(f"   Text: '{text[:50]}...' (Length: {len(text)} chars)")
    print(f"   Style: {style}, Rate: {rate}, Pitch: {pitch}")
    
    # Build the API request
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # All the parameters Murf needs to generate speech
    payload = {
        "voiceId": murf_voice_id,  
        "style": style,  
        "text": text,  
        "rate": rate,  
        "pitch": pitch,  
        "sampleRate": DEFAULT_SAMPLE_RATE,
        "format": DEFAULT_FORMAT, 
        "channelType": "STEREO",  
        "pronunciationDictionary": {},  
        "encodeAsBase64": False  
    }
    
    # ... rest of the function remains the same ...
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"📡 Attempt {attempt}/{MAX_RETRIES}: Calling Murf API...")
            
            response = requests.post(
                MURF_TTS_URL, 
                json=payload, 
                headers=headers, 
                timeout=REQUEST_TIMEOUT
            )
            
            # Success case
            if response.status_code == 200:
                try:
                    response_data = response.json()
                except Exception as e:
                    print(f"❌ ERROR: Failed to parse JSON response: {e}")
                    print(f"   Response text: {response.text[:200]}")
                    return None, None
                
                if 'audioFile' in response_data:
                    audio_url = response_data['audioFile']
                    print(f"📥 Downloading audio from: {audio_url[:50]}...")
                    
                    # Now fetch the actual audio file
                    audio_response = requests.get(audio_url, timeout=REQUEST_TIMEOUT)
                    
                    if audio_response.status_code != 200:
                        print(f"❌ ERROR: Failed to download audio: {audio_response.status_code}")
                        return None, None
                    
                    audio_bytes = audio_response.content
                    
                    if len(audio_bytes) == 0:
                        print("❌ ERROR: Received empty audio response")
                        return None, None
                    
                    print(f"✅ SUCCESS: Generated {len(audio_bytes):,} bytes of audio with voice '{voice_id}'")
                    mime = "audio/mpeg"
                    return audio_bytes, mime
                else:
                    print(f"❌ ERROR: No audioFile in response")
                    return None, None
            
            # Handle non-200 status codes
            else:
                print(f"❌ ERROR: API returned status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
                # Some errors are worth retrying (rate limits, server errors)
                if response.status_code in [429, 500, 502, 503, 504]:
                    if attempt < MAX_RETRIES:
                        wait_time = attempt * 2
                        print(f"🔄 Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                # Other errors (auth, bad request) won't be fixed by retrying
                return None, None
        
        # Handle network-related errors
        except requests.exceptions.Timeout:
            print(f"⏱️ ERROR: Request timeout after {REQUEST_TIMEOUT}s")
            if attempt < MAX_RETRIES:
                print(f"🔄 Retrying...")
                continue
            return None, None
        
        except requests.exceptions.ConnectionError:
            print(f"🌐 ERROR: Connection error. Check your internet connection.")
            if attempt < MAX_RETRIES:
                print(f"🔄 Retrying in {attempt * 2} seconds...")
                time.sleep(attempt * 2)
                continue
            return None, None
        
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Request failed - {str(e)}")
            return None, None
        
        except Exception as e:
            # Catch-all for unexpected errors
            print(f"❌ UNEXPECTED ERROR: {type(e).__name__} - {str(e)}")
            return None, None
    
    print(f"❌ FAILED: All {MAX_RETRIES} attempts exhausted")
    return None, None


def test_murf_api():
    # Simple test function to verify everything is working
    # Run this before submitting to make sure API key is valid
    print("\n" + "="*50)
    print("🧪 Testing Murf API Connection...")
    print("="*50)
    
    test_text = "Hello, this is a test of the Murf text to speech API."
    audio, mime = murf_synthesize(test_text)
    
    if audio:
        print(f"✅ TEST PASSED: Audio generated successfully!")
        print(f"   Size: {len(audio):,} bytes")
        print(f"   Type: {mime}")
        
        # Save test file so you can listen to it
        try:
            with open("test_output.mp3", "wb") as f:
                f.write(audio)
            print(f"💾 Test audio saved as 'test_output.mp3'")
        except Exception as e:
            print(f"⚠️ Could not save test file: {e}")
    else:
        print("❌ TEST FAILED: Could not generate audio")
    
    print("="*50 + "\n")
    return audio is not None


# If you run this file directly (python tts.py), it will test the API
if __name__ == "__main__":
    success = test_murf_api()

    exit(0 if success else 1)


