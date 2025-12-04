import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


def get_asr_status():
    """Check which ASR services are configured"""
    status = {
        "assemblyai": bool(ASSEMBLYAI_API_KEY),
        "deepgram": bool(DEEPGRAM_API_KEY),
        "browser": True  
    }
    return status


def get_preferred_asr():
    """Return the preferred ASR method based on what's configured"""
    # Priority: Deepgram > AssemblyAI > Browser
    if DEEPGRAM_API_KEY:
        return "deepgram"
    elif ASSEMBLYAI_API_KEY:
        return "assemblyai"
    else:
        return "browser"

