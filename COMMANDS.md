# Quick Reference Commands

## Setup Commands

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

## Running the Application

```bash
# Terminal 1 - Start Backend
python backend/app.py

# Terminal 2 - Start Frontend
cd frontend
python -m http.server 3000
```

## Testing Commands

```bash
# Test TTS API
python backend/tts.py

# Test backend health
curl http://localhost:8000/

# Test voices endpoint
curl http://localhost:8000/voices

# Test config endpoint
curl http://localhost:8000/config
```

## Git Commands

```bash
# Check status
git status

# Add all files
git add .

# Commit changes
git commit -m "Final submission ready"

# Push to GitHub
git push origin main

# Check remote
git remote -v

# View commit history
git log --oneline
```

## Troubleshooting Commands

```bash
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list

# Check if port is in use (Windows)
netstat -ano | findstr :8000
netstat -ano | findstr :3000

# Kill process on port (Windows)
# Find PID from netstat, then:
taskkill /PID <pid> /F
```

## File Operations

```bash
# List all files
dir /s /b

# Find specific file
dir /s /b *app.py

# Check file size
dir backend\tts.py

# View file content
type backend\app.py
```

## Virtual Environment

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rmdir /s venv

# Recreate virtual environment
python -m venv venv
```

## Package Management

```bash
# Install specific package
pip install flask

# Upgrade package
pip install --upgrade flask

# Uninstall package
pip uninstall flask

# Freeze requirements
pip freeze > backend/requirements.txt
```

## Quick Tests

```bash
# One-liner to test everything
python backend/tts.py && curl http://localhost:8000/ && curl http://localhost:8000/voices

# Check if backend is running
curl http://localhost:8000/ || echo "Backend not running"

# Check if frontend is accessible
curl http://localhost:3000/ || echo "Frontend not running"
```

## Development Workflow

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Start backend (Terminal 1)
python backend/app.py

# 3. Start frontend (Terminal 2)
cd frontend && python -m http.server 3000

# 4. Open browser
start http://localhost:3000

# 5. Test changes
python backend/tts.py
```

## Submission Workflow

```bash
# 1. Final test
python backend/tts.py

# 2. Commit changes
git add .
git commit -m "Final version for submission"

# 3. Push to GitHub
git push origin main

# 4. Verify on GitHub
start https://github.com/yourusername/yourrepo

# 5. Test clone
cd /tmp
git clone https://github.com/yourusername/yourrepo test
cd test
```

## Useful Shortcuts

```bash
# Clear terminal
cls

# Show current directory
cd

# Go to project root
cd D:\voice_agent

# Open in VS Code
code .

# Open in file explorer
explorer .
```

## Environment Variables

```bash
# Check if .env exists
dir backend\.env

# View .env (be careful!)
type backend\.env

# Create .env from template
echo MURF_API_KEY=your_key_here > backend\.env
```

## Process Management

```bash
# List Python processes
tasklist | findstr python

# Kill all Python processes (careful!)
taskkill /IM python.exe /F

# Check running servers
netstat -ano | findstr LISTENING
```

## Quick Fixes

```bash
# Backend won't start - check port
netstat -ano | findstr :8000

# Frontend won't start - check port
netstat -ano | findstr :3000

# Module not found - reinstall
pip install -r backend/requirements.txt

# API key error - check .env
type backend\.env

# Permission error - run as admin
# Right-click terminal -> Run as Administrator
```

## Demo Preparation

```bash
# 1. Clean up
del test_output.mp3
cls

# 2. Start fresh
python backend/app.py

# 3. Open browser
start http://localhost:3000

# 4. Test voice
# Say: "Hello, what can you do?"
```

## Post-Submission

```bash
# Keep servers running
# Don't close terminals

# Monitor logs
# Watch Terminal 1 (backend) for requests

# Test from different device
# Use your phone to test the URL
```

## Emergency Commands

```bash
# Everything broken? Reset:
deactivate
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/app.py

# Still broken? Check:
python --version
pip --version
type backend\.env
dir backend\
```

## Useful Aliases (Optional)

Add to your shell profile:

```bash
# Quick start backend
alias start-backend="cd D:\voice_agent && venv\Scripts\activate && python backend/app.py"

# Quick start frontend
alias start-frontend="cd D:\voice_agent\frontend && python -m http.server 3000"

# Quick test
alias test-tts="cd D:\voice_agent && venv\Scripts\activate && python backend/tts.py"
```

## Documentation Commands

```bash
# View README
type README.md

# View all markdown files
dir *.md

# Search in files
findstr /s /i "murf" *.md
```

## Final Checklist Commands

```bash
# 1. Test TTS
python backend/tts.py

# 2. Test backend
curl http://localhost:8000/

# 3. Test voices
curl http://localhost:8000/voices

# 4. Check git status
git status

# 5. Check .gitignore
type .gitignore

# 6. Verify no secrets
git grep -i "api_key" -- ':!*.md' ':!.env'
```

---

**Pro Tip:** Keep these commands handy during demo recording and presentation!
