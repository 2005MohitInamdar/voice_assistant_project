"""
Comprehensive test suite to verify all backend functions are working
"""
import os
import sys
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv('.env')

print("=" * 70)
print("COMPREHENSIVE BACKEND FUNCTION TEST")
print("=" * 70)

# Test results tracker
tests_passed = 0
tests_failed = 0
test_results = []

def test_result(name, passed, message=""):
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        status = "✅ PASS"
    else:
        tests_failed += 1
        status = "❌ FAIL"
    
    result = f"{status} - {name}"
    if message:
        result += f": {message}"
    test_results.append(result)
    print(result)

print("\n1. ENVIRONMENT VARIABLES TEST")
print("-" * 70)

# Test 1: Check MURF_API_KEY
murf_key = os.getenv("MURF_API_KEY")
test_result("MURF_API_KEY loaded", murf_key is not None, 
            f"Key starts with: {murf_key[:20] if murf_key else 'N/A'}")

# Test 2: Check GROQ_API_KEY
groq_key = os.getenv("GROQ_API_KEY")
test_result("GROQ_API_KEY loaded", groq_key is not None,
            f"Key starts with: {groq_key[:20] if groq_key else 'N/A'}")

print("\n2. BACKEND API ENDPOINTS TEST")
print("-" * 70)

BASE_URL = "http://127.0.0.1:8000"

# Test 3: Root endpoint
try:
    response = requests.get(f"{BASE_URL}/", timeout=5)
    test_result("Root endpoint (/)", response.status_code == 200,
                f"Status: {response.status_code}")
except Exception as e:
    test_result("Root endpoint (/)", False, str(e))

# Test 4: Config endpoint
try:
    response = requests.get(f"{BASE_URL}/config", timeout=5)
    test_result("Config endpoint (/config)", response.status_code == 200,
                f"Status: {response.status_code}")
except Exception as e:
    test_result("Config endpoint (/config)", False, str(e))

# Test 5: Voices endpoint
try:
    response = requests.get(f"{BASE_URL}/voices", timeout=5)
    voices_ok = response.status_code == 200
    if voices_ok:
        voices = response.json()
        voices_ok = len(voices) == 8
        test_result("Voices endpoint (/voices)", voices_ok,
                    f"Found {len(voices)} voices")
    else:
        test_result("Voices endpoint (/voices)", False,
                    f"Status: {response.status_code}")
except Exception as e:
    test_result("Voices endpoint (/voices)", False, str(e))

print("\n3. TTS FUNCTIONALITY TEST")
print("-" * 70)

# Test 6: Import tts module
try:
    from tts import murf_synthesize, get_available_voices
    test_result("TTS module import", True, "Successfully imported")
except Exception as e:
    test_result("TTS module import", False, str(e))
    sys.exit(1)

# Test 7: Get available voices
try:
    voices = get_available_voices()
    test_result("Get available voices", len(voices) == 8,
                f"Found {len(voices)} voices")
except Exception as e:
    test_result("Get available voices", False, str(e))

# Test 8: Test male voice (Wayne)
try:
    audio, mime = murf_synthesize("Testing male voice", voice_id="en-US-wayne")
    test_result("Male voice synthesis (Wayne)", 
                audio is not None and len(audio) > 0,
                f"Generated {len(audio) if audio else 0} bytes")
except Exception as e:
    test_result("Male voice synthesis (Wayne)", False, str(e))

# Test 9: Test female voice (Natalie)
try:
    audio, mime = murf_synthesize("Testing female voice", voice_id="en-US-natalie")
    test_result("Female voice synthesis (Natalie)", 
                audio is not None and len(audio) > 0,
                f"Generated {len(audio) if audio else 0} bytes")
except Exception as e:
    test_result("Female voice synthesis (Natalie)", False, str(e))

# Test 10: Test another female voice (Samantha)
try:
    audio, mime = murf_synthesize("Testing another female voice", voice_id="en-US-samantha")
    test_result("Female voice synthesis (Samantha)", 
                audio is not None and len(audio) > 0,
                f"Generated {len(audio) if audio else 0} bytes")
except Exception as e:
    test_result("Female voice synthesis (Samantha)", False, str(e))

print("\n4. LLM INTEGRATION TEST")
print("-" * 70)

# Test 11: Import logic module
try:
    from logic import generate_reply
    test_result("Logic module import", True, "Successfully imported")
except Exception as e:
    test_result("Logic module import", False, str(e))

# Test 12: Generate reply
try:
    reply = generate_reply("Hello, how are you?")
    test_result("LLM reply generation", 
                reply is not None and len(reply) > 0 and "trouble" not in reply.lower(),
                f"Reply: {reply[:50]}...")
except Exception as e:
    test_result("LLM reply generation", False, str(e))

print("\n5. END-TO-END INTEGRATION TEST")
print("-" * 70)

# Test 13: Full synthesize endpoint
try:
    response = requests.post(
        f"{BASE_URL}/synthesize",
        json={
            "text": "Hello, this is a test",
            "voiceId": "en-US-cooper",
            "style": "Conversational",
            "rate": 0,
            "pitch": 0
        },
        timeout=30
    )
    audio_ok = response.status_code == 200 and len(response.content) > 0
    test_result("Full synthesize endpoint", audio_ok,
                f"Status: {response.status_code}, Audio: {len(response.content)} bytes")
except Exception as e:
    test_result("Full synthesize endpoint", False, str(e))

print("\n6. ERROR HANDLING TEST")
print("-" * 70)

# Test 14: Invalid voice ID handling
try:
    audio, mime = murf_synthesize("Test", voice_id="invalid-voice-id")
    # Should return None for invalid voice
    test_result("Invalid voice ID handling", audio is None,
                "Properly handled invalid voice")
except Exception as e:
    test_result("Invalid voice ID handling", False, str(e))

# Test 15: Empty text handling
try:
    audio, mime = murf_synthesize("", voice_id="en-US-wayne")
    test_result("Empty text handling", audio is None,
                "Properly handled empty text")
except Exception as e:
    test_result("Empty text handling", False, str(e))

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests: {tests_passed + tests_failed}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
print(f"Success Rate: {(tests_passed/(tests_passed+tests_failed)*100):.1f}%")
print("=" * 70)

if tests_failed == 0:
    print("\n🎉 ALL TESTS PASSED! System is fully functional.")
    sys.exit(0)
else:
    print(f"\n⚠️ {tests_failed} test(s) failed. Please review the issues above.")
    sys.exit(1)
