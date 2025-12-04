# Pre-Flight Checklist ✈️

## Run This Before Submitting!

### 1. Environment Check
```bash
# Verify Python version
python --version  # Should be 3.8+

# Verify virtual environment is activated
# You should see (venv) in your terminal prompt
```

### 2. Backend Health Check
```bash
# Test TTS API
python backend/tts.py
# Expected: "TEST PASSED: Audio generated successfully!"

# Start backend
python backend/app.py
# Expected: Running on http://0.0.0.0:8000
```

### 3. API Endpoints Check
Open new terminal:
```bash
# Test health endpoint
curl http://localhost:8000/
# Expected: "Voice Agent Backend Running"

# Test voices endpoint
curl http://localhost:8000/voices
# Expected: JSON array with 8 voices

# Test config endpoint
curl http://localhost:8000/config
# Expected: JSON with ASR configuration
```

### 4. Frontend Check
```bash
# Start frontend
cd frontend
python -m http.server 3000
# Expected: Serving HTTP on 0.0.0.0 port 3000
```

### 5. Browser Test
1. Open http://localhost:3000 in Chrome
2. Check console (F12) - should have no errors
3. Status should show "Backend connected"
4. Voice dropdown should show 8 voices

### 6. End-to-End Test
1. Click "Start Recording"
2. Say: "Hello, what time is it?"
3. Wait for response
4. Audio should play automatically
5. Try different voice
6. Try adjusting rate/pitch
7. Test again

### 7. File Verification

#### Check these files exist:
- [ ] README.md
- [ ] QUICKSTART.md
- [ ] DEMO_SCRIPT.md
- [ ] LINKEDIN_POST.md
- [ ] TESTING_CHECKLIST.md
- [ ] SUBMISSION_GUIDE.md
- [ ] PROJECT_SUMMARY.md
- [ ] .gitignore
- [ ] backend/app.py
- [ ] backend/tts.py
- [ ] backend/logic.py
- [ ] backend/asr.py
- [ ] backend/requirements.txt
- [ ] backend/.env (NOT in git!)
- [ ] frontend/index.html
- [ ] frontend/app.jsx

#### Check .gitignore includes:
- [ ] .env
- [ ] backend/.env
- [ ] __pycache__/
- [ ] venv/
- [ ] *.mp3

### 8. Security Check
```bash
# Make sure no API keys in git
git grep -i "api_key" -- ':!*.md' ':!.env'
# Should return nothing or only comments

# Check .env is ignored
git status
# .env should NOT appear in untracked files
```

### 9. Code Quality Check
- [ ] All files have proper comments
- [ ] No console.log() left in production code
- [ ] No TODO comments left
- [ ] No hardcoded values that should be configurable
- [ ] All functions have docstrings (Python)

### 10. Documentation Check
- [ ] README has setup instructions
- [ ] README has demo video link placeholder
- [ ] README mentions Murf Falcon
- [ ] All markdown files are readable
- [ ] No broken links in documentation

### 11. Demo Video Preparation
- [ ] Script prepared (DEMO_SCRIPT.md)
- [ ] Screen recording software ready
- [ ] Microphone tested
- [ ] Application working smoothly
- [ ] No sensitive data visible

### 12. LinkedIn Post Preparation
- [ ] Template reviewed (LINKEDIN_POST.md)
- [ ] Includes "@Murf AI"
- [ ] Includes "Built using Murf Falcon – the consistently fastest TTS API."
- [ ] GitHub link ready
- [ ] Demo video link ready (after recording)

### 13. GitHub Repository Check
```bash
# Check remote is set
git remote -v

# Check all files are committed
git status
# Should show "nothing to commit, working tree clean"

# Check branch
git branch
# Should be on main or master

# Push to GitHub
git push origin main
```

### 14. Repository Settings
On GitHub:
- [ ] Repository is public
- [ ] Description added
- [ ] Topics include "murf-ai"
- [ ] README displays correctly
- [ ] All files visible

### 15. Final Test from Fresh Clone
```bash
# Clone in new directory
cd /tmp
git clone <your-repo-url> test-clone
cd test-clone

# Follow README setup instructions
# Verify everything works
```

### 16. Submission Form Data Ready
Have these ready:
- [ ] Team name
- [ ] Team member names
- [ ] GitHub repository URL
- [ ] LinkedIn post URL (after posting)
- [ ] Demo video URL (after uploading)
- [ ] Contact email

### 17. Timing Check
- [ ] Current date/time noted
- [ ] Deadline confirmed: Dec 5, 2025, 11:59 PM
- [ ] Time zone confirmed
- [ ] Buffer time planned (submit 2+ hours early)

### 18. Backup Plan
- [ ] Code backed up locally
- [ ] Demo video backed up
- [ ] Screenshots taken
- [ ] Links saved in document

## If Everything Passes ✅

You're ready to:
1. Record demo video
2. Upload video
3. Update README with video link
4. Push to GitHub
5. Create LinkedIn post
6. Submit form

## If Something Fails ❌

### Backend won't start
- Check Python version
- Check virtual environment activated
- Check requirements installed
- Check .env file exists
- Check port 8000 not in use

### Frontend won't load
- Check port 3000 not in use
- Check index.html exists
- Check app.jsx exists
- Try different browser

### TTS test fails
- Check API key in .env
- Check internet connection
- Check API key is valid
- Try running test again

### Speech recognition not working
- Use Chrome browser
- Allow microphone permission
- Check microphone is working
- Try in incognito mode

### Audio not playing
- Check browser audio settings
- Check audio file downloaded
- Try clicking play manually
- Check browser console for errors

## Emergency Contacts

- Murf Discord: [Link]
- Techfest Email: vaibhav@techfest.org
- Teammate: [Your teammate contact]

## Final Confidence Check

Rate your confidence (1-10) on:
- [ ] Application works: ___/10
- [ ] Documentation complete: ___/10
- [ ] Demo video ready: ___/10
- [ ] LinkedIn post ready: ___/10
- [ ] Submission form ready: ___/10

**If all above 8/10, you're good to go!** 🚀

## Post-Submission

After submitting:
- [ ] Save confirmation email/screenshot
- [ ] Note submission timestamp
- [ ] Keep application running for testing
- [ ] Monitor LinkedIn post engagement
- [ ] Prepare for potential finalist presentation

---

**Good luck! You've got this!** 💪
