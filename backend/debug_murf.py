import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

print("\n" + "="*70)
print("🔍 MURF API COMPLETE DIAGNOSTIC")
print("="*70)

# Check API Key
MURF_API_KEY = os.getenv("MURF_API_KEY")

print("\n📋 STEP 1: API Key Check")
print("-" * 70)
if not MURF_API_KEY:
    print("❌ CRITICAL: MURF_API_KEY not found!")
    print("💡 Check your .env file")
    exit(1)
else:
    print(f"✅ API Key Found")
    print(f"   First 15 chars: {MURF_API_KEY[:15]}...")
    print(f"   Last 10 chars: ...{MURF_API_KEY[-10:]}")
    print(f"   Total length: {len(MURF_API_KEY)} characters")

# Test 1: Current endpoint with detailed logging
print("\n📋 STEP 2: Testing Current Endpoint")
print("-" * 70)

url = "https://api.murf.ai/v1/speech/stream"
headers = {
    "Authorization": f"Bearer {MURF_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": "Hello, this is a test.",
    "model": "falcon",
    "voice": "alloy",
    "format": "mp3",
    "channelType": "MONO"
}

print(f"URL: {url}")
print(f"Headers: {json.dumps({k: v[:20]+'...' if k == 'Authorization' else v for k, v in headers.items()}, indent=2)}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    
    print(f"\n📥 Response:")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response Headers: {json.dumps(dict(response.headers), indent=2)}")
    print(f"   Content Length: {len(response.content)} bytes")
    print(f"   Content Type: {response.headers.get('Content-Type', 'N/A')}")
    
    if response.status_code == 200:
        print(f"\n✅ SUCCESS!")
        with open("test_success.mp3", "wb") as f:
            f.write(response.content)
        print(f"💾 Audio saved as test_success.mp3")
    else:
        print(f"\n❌ ERROR Response:")
        print(f"   {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out after 15 seconds")
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")

# Test 2: Try alternative endpoints
print("\n📋 STEP 3: Testing Alternative Endpoints")
print("-" * 70)

alternative_endpoints = [
    "https://api.murf.ai/v1/speech/generate",
    "https://api.murf.ai/v1/speech/generate-with-audio",
    "https://api.murf.ai/v1/tts",
    "https://api.murf.ai/v2/speech/stream",
    "https://api.murf.ai/speech/stream",
]

for endpoint in alternative_endpoints:
    print(f"\n🔄 Testing: {endpoint}")
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ THIS ENDPOINT WORKS!")
            print(f"   Content: {len(response.content)} bytes")
        else:
            print(f"   Response: {response.text[:150]}")
    except Exception as e:
        print(f"   ❌ {type(e).__name__}: {str(e)[:100]}")

# Test 3: Try different authentication methods
print("\n📋 STEP 4: Testing Different Auth Methods")
print("-" * 70)

auth_methods = [
    {"Authorization": f"Bearer {MURF_API_KEY}"},
    {"Authorization": f"Token {MURF_API_KEY}"},
    {"x-api-key": MURF_API_KEY},
    {"api-key": MURF_API_KEY},
    {"X-API-Key": MURF_API_KEY},
]

url = "https://api.murf.ai/v1/speech/stream"

for i, auth_header in enumerate(auth_methods, 1):
    print(f"\n🔑 Auth method {i}: {list(auth_header.keys())[0]}")
    headers_test = {**auth_header, "Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers_test, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ THIS AUTH METHOD WORKS!")
        elif response.status_code == 401:
            print(f"   ❌ 401 Unauthorized - Wrong auth method or invalid key")
        else:
            print(f"   Response: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")

# Test 4: Check if API key can access account info
print("\n📋 STEP 5: Testing Account/Voice Endpoints")
print("-" * 70)

info_endpoints = [
    "https://api.murf.ai/v1/voices",
    "https://api.murf.ai/v1/account",
    "https://api.murf.ai/v1/user",
    "https://api.murf.ai/v1/projects",
]

headers = {"Authorization": f"Bearer {MURF_API_KEY}"}

for endpoint in info_endpoints:
    print(f"\n📡 GET {endpoint}")
    try:
        response = requests.get(endpoint, headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ SUCCESS! Response: {response.text[:200]}")
        else:
            print(f"   Response: {response.text[:150]}")
    except Exception as e:
        print(f"   ❌ {type(e).__name__}")

print("\n" + "="*70)
print("🏁 DIAGNOSTIC COMPLETE")
print("="*70)
print("\n📊 SUMMARY:")
print("   Copy the entire output above and share it")
print("   This will help identify the exact issue")
print("="*70 + "\n")