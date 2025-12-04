# Testing Checklist

## Pre-Submission Testing

### Backend Tests

- [x] API key loads correctly from .env
- [x] TTS API connection works (run `python backend/tts.py`)
- [x] Flask server starts without errors
- [x] All endpoints respond correctly:
  - [ ] GET / (health check)
  - [ ] GET /voices (returns voice list)
  - [ ] GET /config (returns ASR config)
  - [ ] POST /synthesize (generates audio)

### Frontend Tests

- [ ] Page loads without console errors
- [ ] Backend connection status shows "Backend connected"
- [ ] Voice dropdown populates with 8 voices
- [ ] Style dropdown shows all 6 styles
- [ ] Speech recognition starts when clicking "Start Recording"
- [ ] Transcript updates in real-time while speaking
- [ ] Audio response plays automatically
- [ ] Audio player controls work
- [ ] Settings persist during session

### Voice Features

Test each voice:
- [ ] Ken (Male, US, Professional)
- [ ] Natalie (Female, US, Friendly)
- [ ] Cooper (Male, US, Casual)
- [ ] Sarah (Female, US, Warm)
- [ ] Michael (Male, US, Authoritative)
- [ ] Lisa (Female, US, Energetic)
- [ ] Charles (Male, UK, Formal)
- [ ] Emma (Female, UK, Elegant)

### Conversation Tests

Try these phrases:
- [ ] "Hello" - Should greet back
- [ ] "What time is it?" - Should tell current time
- [ ] "What's the date?" - Should tell current date
- [ ] "What day is it?" - Should tell day of week
- [ ] "What can you do?" - Should list capabilities
- [ ] "Tell me about Murf" - Should describe Murf Falcon
- [ ] "Thank you" - Should respond politely
- [ ] "Goodbye" - Should say goodbye

### Advanced Settings

- [ ] Speech rate slider works (-50 to 50)
- [ ] Pitch slider works (-50 to 50)
- [ ] Changes apply to next speech generation
- [ ] Advanced settings panel toggles correctly

### Error Handling

- [ ] Microphone permission denied - Shows helpful error
- [ ] No speech detected - Shows appropriate message
- [ ] Backend offline - Shows connection error
- [ ] Invalid voice ID - Falls back to default
- [ ] Empty speech - Handles gracefully

### Performance

- [ ] Response time < 3 seconds for typical phrase
- [ ] No memory leaks after multiple uses
- [ ] Audio plays smoothly without stuttering
- [ ] UI remains responsive during processing

### Browser Compatibility

- [ ] Chrome (recommended)
- [ ] Edge
- [ ] Firefox (limited speech recognition)
- [ ] Safari (limited speech recognition)

### Keyboard Shortcuts

- [ ] Space bar starts/stops recording
- [ ] Works when not focused on input fields

## Demo Video Checklist

- [ ] Show clean interface
- [ ] Demonstrate voice interaction
- [ ] Show different voices
- [ ] Show advanced settings
- [ ] Explain technical stack
- [ ] Mention Murf Falcon
- [ ] Show code structure briefly
- [ ] Keep under 3 minutes

## GitHub Checklist

- [ ] README.md is complete
- [ ] .gitignore includes .env
- [ ] No API keys in repository
- [ ] requirements.txt is accurate
- [ ] Demo video link added
- [ ] Repository tagged with `murf-ai`
- [ ] Code is well-commented
- [ ] Project structure is clear

## LinkedIn Post Checklist

- [ ] Project description included
- [ ] @Murf AI tagged
- [ ] Includes: "Built using Murf Falcon – the consistently fastest TTS API."
- [ ] GitHub link included
- [ ] Demo video link included
- [ ] Relevant hashtags added
- [ ] Posted before deadline

## Final Checks

- [ ] All files saved
- [ ] Backend runs without errors
- [ ] Frontend loads correctly
- [ ] End-to-end test successful
- [ ] Demo video recorded
- [ ] LinkedIn post drafted
- [ ] Submission form ready
