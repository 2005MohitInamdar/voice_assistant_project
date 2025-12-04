import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import './index.css';
import EdgeParticles from './components/ParticleBackground'  
function App() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState('Ready');
  const [statusType, setStatusType] = useState('idle');
  const [audioUrl, setAudioUrl] = useState(null);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('en-US-ken');
  const [selectedStyle, setSelectedStyle] = useState('Conversational');
  const [rate, setRate] = useState(0);
  const [pitch, setPitch] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isLoadingVoices, setIsLoadingVoices] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [textInput, setTextInput] = useState('');
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);
  const settingsRef = useRef({ 
    voice: 'en-US-ken', 
    style: 'Conversational', 
    rate: 0, 
    pitch: 0 
  });
  const isProcessingRef = useRef(false);
  const processedTextRef = useRef('');
  const safeZone = {
  x: window.innerWidth / 2 - 300, // approximate left of your box
  y: window.innerHeight / 2 - 300, // top
  width: 900, // width of your inner box
  height: 600 // height of your inner box
};
  useEffect(() => {
    setIsClient(true);
  }, []);
  // Clean up audio URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Initialize speech recognition
  useEffect(() => {
    let recognition = null;
    
    if ('webkitSpeechRecognition' in window) {
      recognition = new window.webkitSpeechRecognition();
    } else if ('SpeechRecognition' in window) {
      recognition = new window.SpeechRecognition();
    } else {
      setStatus('Speech Recognition not supported in this browser. Use Chrome.');
      setStatusType('error');
      return;
    }

    recognition.lang = 'en-US';
    recognition.continuous = true;
    recognition.interimResults = true;
    recognitionRef.current = recognition;

    // Setup recognition event handlers
    // const handleRecognitionResult = async (event) => {
    //   let finalText = '';
    //   let interimText = '';

    //   for (let i = event.resultIndex; i < event.results.length; i++) {
    //     const transcript = event.results[i][0].transcript;
    //     if (event.results[i].isFinal) {
    //       finalText += transcript + ' ';
    //     } else {
    //       interimText += transcript;
    //     }
    //   }

    //   // Show interim results
    //   if (interimText) {
    //     setTranscript(interimText);
    //   }

    //   // Only process final results
    //   if (finalText.trim() === '') return;
      
    //   // Prevent duplicate processing
    //   if (isProcessingRef.current || processedTextRef.current === finalText.trim()) {
    //     return;
    //   }

    //   isProcessingRef.current = true;
    //   processedTextRef.current = finalText.trim();
      
    //   // Stop recognition temporarily
    //   recognition.stop();
    //   setIsListening(false);

    //   setTranscript(finalText.trim());
    //   setStatus('Processing speech...');
    //   setStatusType('info');

    //   await processAndSynthesizeSpeech(finalText.trim());
    // };


  const handleRecognitionResult = async (event) => {
    let finalText = '';
    let interimText = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += transcript + ' ';
      } else {
        interimText += transcript;
      }
    }

    if (interimText) {
      setTranscript(interimText);
    }

    if (finalText.trim() === '') return;

    if (isProcessingRef.current || processedTextRef.current === finalText.trim()) {
      return;
    }

    isProcessingRef.current = true;
    processedTextRef.current = finalText.trim();

    const userMessage = finalText.trim();

    // ADD USER MESSAGE TO SIDEBAR IMMEDIATELY
    setMessages(prev => [...prev, { role: "user", text: userMessage }]);

    recognition.stop();
    setIsListening(false);
    setTranscript(userMessage);
    setStatus('Thinking...');
    setStatusType('loading');

    await processAndSynthesizeSpeech(userMessage);
  };

    const handleRecognitionError = (event) => {
      console.error('Recognition error:', event.error);
      
      switch (event.error) {
        case 'no-speech':
          setStatus('No speech detected. Please try again.');
          break;
        case 'audio-capture':
          setStatus('Microphone not accessible. Please check permissions.');
          break;
        case 'not-allowed':
          setStatus('Microphone permission denied. Please allow microphone access.');
          break;
        case 'aborted':
          // Ignore aborted errors (happens when we stop manually)
          return;
        default:
          setStatus(`Error: ${event.error}`);
      }
      
      setStatusType('error');
      setIsListening(false);
      isProcessingRef.current = false;
    };

    const handleRecognitionEnd = () => {
      if (!isProcessingRef.current && isListening) {
        setIsListening(false);
        setStatus('Recognition ended. Click Start to try again.');
        setStatusType('idle');
      }
    };

    recognition.onresult = handleRecognitionResult;
    recognition.onerror = handleRecognitionError;
    recognition.onend = handleRecognitionEnd;

    // Cleanup function
    return () => {
      if (recognition) {
        recognition.stop();
        recognition.onresult = null;
        recognition.onerror = null;
        recognition.onend = null;
      }
    };
  }, []);

  // Fetch voices and check backend connection
  useEffect(() => {
    const initializeApp = async () => {
      setIsLoadingVoices(true);
      try {
        // Test backend connection
        const testResponse = await fetch('http://127.0.0.1:8000/');
        if (!testResponse.ok) {
          throw new Error('Backend connection failed');
        }
        
        setStatus('Backend connected');
        setStatusType('success');
        
        // Fetch available voices
        const voicesResponse = await fetch('http://127.0.0.1:8000/voices');
        if (!voicesResponse.ok) {
          throw new Error('Failed to fetch voices');
        }
        
        const voicesData = await voicesResponse.json();
        setVoices(voicesData);
        
        if (voicesData.length > 0) {
          setSelectedVoice(voicesData[0].id);
          settingsRef.current.voice = voicesData[0].id;
        }
      } catch (error) {
        console.error('Initialization error:', error);
        setStatus('Backend not reachable. Make sure it is running on port 8000.');
        setStatusType('error');
      } finally {
        setIsLoadingVoices(false);
      }
    };

    initializeApp();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Space bar to start/stop recording (only when not interacting with form elements)
      if (e.code === 'Space' && 
          e.target.tagName !== 'TEXTAREA' && 
          e.target.tagName !== 'INPUT' &&
          e.target.tagName !== 'BUTTON' &&
          e.target.tagName !== 'SELECT' &&
          !e.target.isContentEditable) {
        e.preventDefault();
        
        if (isListening) {
          stopListening();
        } else {
          startListening();
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isListening]);

  // const processAndSynthesizeSpeech = useCallback(async (text) => {
  //   try {
  //     setStatus('Processing with AI...');
  //     setStatusType('loading');
      
  //     const response = await fetch('http://127.0.0.1:8000/synthesize', {
  //       method: 'POST',
  //       headers: { 'Content-Type': 'application/json' },
  //       body: JSON.stringify({ 
  //         text: text,
  //         voiceId: settingsRef.current.voice,
  //         style: settingsRef.current.style,
  //         rate: settingsRef.current.rate,
  //         pitch: settingsRef.current.pitch
  //       })
  //     });

  //     if (!response.ok) {
  //       const errorText = await response.text();
  //       throw new Error(`Server error: ${response.status} - ${errorText}`);
  //     }

  //     const arrayBuffer = await response.arrayBuffer();
  //     if (arrayBuffer.byteLength === 0) {
  //       throw new Error('Received empty audio response');
  //     }
      
  //     // Clean up previous audio URL if exists
  //     if (audioUrl) {
  //       URL.revokeObjectURL(audioUrl);
  //     }
  //     const blob = new Blob([arrayBuffer], { type: 'audio/mpeg' });
  //     const url = URL.createObjectURL(blob);


      
  //     setAudioUrl(url);


  //     // ADD AI REPLY HERE
  //   setTimeout(() => {
  //     setMessages(prev => [...prev, { 
  //       role: "assistant", 
  //       text: "This is a beautiful voice AI agent built with Murf and Groq!" // Replace with real reply later
  //     }]);

  //     // Auto-scroll sidebar
  //     const panel = document.getElementById('memory-panel');
  //     if (panel) panel.scrollTop = panel.scrollHeight;
  //   }, 1000);


  //     setStatus('Playing response...');
  //     setStatusType('success');
      
  //     // Wait for next render to ensure audio element has the new src
  //     await new Promise(resolve => setTimeout(resolve, 100));
      
  //     if (audioRef.current) {
  //       // Setup audio event handlers
  //       const audioElement = audioRef.current;
        
  //       const handleAudioEnd = () => {
  //         setStatus('Response complete. Click Start to continue.');
  //         setStatusType('idle');
  //         audioElement.removeEventListener('ended', handleAudioEnd);
  //         audioElement.removeEventListener('error', handleAudioError);
  //       };
        
  //       const handleAudioError = (e) => {
  //         console.error('Audio playback error:', e);
  //         setStatus('Audio playback error. Please try again.');
  //         setStatusType('error');
  //         audioElement.removeEventListener('ended', handleAudioEnd);
  //         audioElement.removeEventListener('error', handleAudioError);
  //       };
        
  //       audioElement.addEventListener('ended', handleAudioEnd);
  //       audioElement.addEventListener('error', handleAudioError);
        
  //       audioElement.src = url;
        
  //       try {
  //         await audioElement.play();
  //       } catch (playError) {
  //         console.error('Audio play error:', playError);
  //         setStatus('Audio loaded. Click play button to hear response.');
  //         setStatusType('success');
  //       }
  //     }
  //   } catch (err) {
  //     console.error('Processing error:', err);
  //     setStatus(`Error: ${err.message}`);
  //     setStatusType('error');
  //   } finally {
  //     isProcessingRef.current = false;
  //   }
  // }, [audioUrl]);




  const processAndSynthesizeSpeech = useCallback(async (text) => {
  try {
    setStatus('Generating response...');
    setStatusType('loading');

    const response = await fetch('http://127.0.0.1:8000/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        text: text,
        voiceId: settingsRef.current.voice,
        style: settingsRef.current.style,
        rate: settingsRef.current.rate,
        pitch: settingsRef.current.pitch
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();  // Now we get JSON!

    // REAL AI REPLY TEXT FROM BACKEND
    const aiReply = data.reply_text || "No reply received";

    // ADD REAL AI MESSAGE TO SIDEBAR
    setMessages(prev => [...prev, { 
      role: "assistant", 
      text: aiReply 
    }]);

    // Auto-scroll
    setTimeout(() => {
      const panel = document.getElementById('memory-panel');
      if (panel) panel.scrollTop = panel.scrollHeight;
    }, 100);

    // Play audio from base64
    const audioBytes = base64ToArrayBuffer(data.audio_base64);
    const blob = new Blob([audioBytes], { type: data.mime_type || 'audio/mpeg' });
    const url = URL.createObjectURL(blob);

    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(url);

    setStatus('Playing response...');
    setStatusType('success');

    await new Promise(resolve => setTimeout(resolve, 100));
    
    if (audioRef.current) {
      audioRef.current.src = url;
      await audioRef.current.play();
    }

  } catch (err) {
    console.error('Processing error:', err);
    setStatus(`Error: ${err.message}`);
    setStatusType('error');
  } finally {
    isProcessingRef.current = false;
  }
}, [audioUrl]);

// Helper function to convert base64 to ArrayBuffer
function base64ToArrayBuffer(base64) {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}


  const [messages, setMessages] = useState([
    // { role: "user" | "assistant", text: "..." }
  ]);

  const handleTextSubmit = async () => {
    const userMessage = textInput.trim();
    if (!userMessage || isProcessingRef.current) return;

    isProcessingRef.current = true;
    
    // Add user message to sidebar
    setMessages(prev => [...prev, { role: "user", text: userMessage }]);
    
    // Clear input and update status
    setTextInput('');
    setTranscript(userMessage);
    setStatus('Processing...');
    setStatusType('loading');

    await processAndSynthesizeSpeech(userMessage);
  };

  const startListening = useCallback(() => {
    if (!recognitionRef.current) {
      setStatus('Speech Recognition not supported');
      setStatusType('error');
      return;
    }
    
    // Reset processing flags
    isProcessingRef.current = false;
    processedTextRef.current = '';
    
    setTranscript('');
    setStatus('Listening... Speak now');
    setStatusType('listening');
    setIsListening(true);
    
    try {
      recognitionRef.current.start();
    } catch (e) {
      console.error('Recognition start error:', e);
      if (e.name === 'InvalidStateError') {
        // Recognition might already be running, try to restart
        recognitionRef.current.stop();
        setTimeout(() => {
          recognitionRef.current.start();
        }, 100);
      } else {
        setStatus('Failed to start recognition');
        setStatusType('error');
        setIsListening(false);
      }
    }
  }, []);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        console.error('Recognition stop error:', e);
      }
    }
    setStatus('Stopped listening');
    setStatusType('idle');
    setIsListening(false);
  }, [isListening]);

  const handleVoiceChange = useCallback((e) => {
    const value = e.target.value;
    setSelectedVoice(value);
    settingsRef.current.voice = value;
  }, []);

  const handleStyleChange = useCallback((e) => {
    const value = e.target.value;
    setSelectedStyle(value);
    settingsRef.current.style = value;
  }, []);

  const handleRateChange = useCallback((e) => {
    const value = parseInt(e.target.value);
    setRate(value);
    settingsRef.current.rate = value;
  }, []);

  const handlePitchChange = useCallback((e) => {
    const value = parseInt(e.target.value);
    setPitch(value);
    settingsRef.current.pitch = value;
  }, []);

  const getStatusColor = () => {
    switch (statusType) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'listening': return 'text-blue-400 animate-pulse';
      case 'loading': return 'text-yellow-400';
      case 'info': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  };

  const styles = ['Conversational', 'Professional', 'Friendly', 'Casual', 'Formal', 'Energetic'];

  return (
    <div className="min-h-screen flex items-center justify-center p-4 md:p-6 bg-gradient-to-b from-blue-950 via-purple-950 to-black">
      <EdgeParticles count={150} safeZone={safeZone}></EdgeParticles>
      <div className="max-w-4xl w-full">
        <div className="bg-gradient-to-r from-purple-900/30 via-blue-900/30 to-purple-900/20 backdrop-blur-2xl rounded-3xl shadow-[0_0_50px_rgba(128,0,255,0.4)] p-6 md:p-8 border border-purple-400/20">

          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-3xl md:text-4xl font-extrabold text-white mb-2 drop-shadow-lg">
              AI Voice Agent by Murf AI
            </h1>
            <p className="text-blue-300/80 text-sm md:text-base">
              Advanced voice synthesis with customizable settings
            </p>
            <p className="text-purple-300 text-xs md:text-sm mt-2">
              Tip: Press Space to start/stop recording
            </p>
          </div>

          {/* Status */}
          <div className="mb-6 p-4 bg-purple-900/40 rounded-xl border border-purple-400/20">
            <div className="flex items-center justify-between">
              <p className={`text-sm md:text-base font-medium ${getStatusColor()} transition-colors`}>
                {status}
              </p>
              {isLoadingVoices && (
                <span className="text-xs text-yellow-400 animate-pulse">
                  Loading voices...
                </span>
              )}
            </div>
          </div>

          {/* Voice Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-blue-300 mb-2">
                Voice {isLoadingVoices && '(Loading...)'}
              </label>
              <select
                value={selectedVoice}
                onChange={handleVoiceChange}
                disabled={isLoadingVoices || voices.length === 0}
                className="w-full px-4 py-3 bg-purple-800/40 border border-purple-400/30 rounded-xl text-blue-100 focus:outline-none focus:ring-2 focus:ring-purple-500 shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {voices.length === 0 ? (
                  <option value="">No voices available</option>
                ) : (
                  voices.map((voice) => (
                    <option key={voice.id} value={voice.id}>
                      {voice.name} ({voice.gender}) - {voice.language}
                    </option>
                  ))
                )}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-blue-300 mb-2">
                Style
              </label>
              <select
                value={selectedStyle}
                onChange={handleStyleChange}
                className="w-full px-4 py-3 bg-purple-800/40 border border-purple-400/30 rounded-xl text-blue-100 focus:outline-none focus:ring-2 focus:ring-purple-500 shadow-md"
              >
                {styles.map(style => (
                  <option key={style} value={style} className="bg-purple-900 text-blue-100">
                    {style}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 mb-4 text-blue-400 hover:text-purple-300 text-sm font-medium transition-colors drop-shadow-lg"
            type="button"
          >
            <span className="text-lg">{showAdvanced ? '▼' : '▶'}</span>
            Advanced Settings
          </button>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6 p-4 bg-purple-900/40 rounded-xl border border-purple-400/20 shadow-inner">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-medium text-blue-300">
                    Speech Rate
                  </label>
                  <span className="text-sm text-purple-300">{rate}</span>
                </div>
                <input
                  type="range"
                  min="-50"
                  max="50"
                  value={rate}
                  onChange={handleRateChange}
                  className="w-full h-2 bg-purple-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
                <div className="flex justify-between text-xs text-purple-300 mt-2">
                  <span>Slower</span>
                  <span>Normal</span>
                  <span>Faster</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-medium text-blue-300">
                    Pitch
                  </label>
                  <span className="text-sm text-purple-300">{pitch}</span>
                </div>
                <input
                  type="range"
                  min="-50"
                  max="50"
                  value={pitch}
                  onChange={handlePitchChange}
                  className="w-full h-2 bg-purple-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
                <div className="flex justify-between text-xs text-purple-300 mt-2">
                  <span>Lower</span>
                  <span>Normal</span>
                  <span>Higher</span>
                </div>
              </div>
            </div>
          )}

          {/* Transcript */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-blue-300 mb-2">
              Transcript
            </label>
            <div className="relative">
              <textarea
                value={transcript}
                readOnly
                placeholder={isListening ? "Listening... speak now" : "Your speech will appear here..."}
                className="w-full h-32 px-4 py-3 bg-purple-800/40 border border-purple-400/30 rounded-xl text-blue-100 placeholder-blue-200/50 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none shadow-inner"
              />
              {transcript && (
                <button
                  onClick={() => setTranscript('')}
                  className="absolute top-2 right-2 p-1 text-xs text-purple-300 hover:text-white bg-purple-900/50 rounded"
                  type="button"
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Text Input for Type-to-Audio */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-blue-300 mb-2">
              Or Type Your Message
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && textInput.trim()) {
                    handleTextSubmit();
                  }
                }}
                placeholder="Type your message here..."
                className="flex-1 px-4 py-3 bg-purple-800/40 border border-purple-400/30 rounded-xl text-blue-100 placeholder-blue-200/50 focus:outline-none focus:ring-2 focus:ring-purple-500 shadow-inner"
              />
              <button
                onClick={handleTextSubmit}
                disabled={!textInput.trim() || isProcessingRef.current}
                className={`px-6 py-3 rounded-xl font-semibold text-white transition-all duration-200 ${
                  !textInput.trim() || isProcessingRef.current
                    ? 'bg-purple-700 cursor-not-allowed opacity-50'
                    : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 shadow-lg hover:shadow-xl active:scale-95'
                }`}
                type="button"
              >
                Send
              </button>
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <button
              onClick={startListening}
              disabled={isListening || isLoadingVoices}
              className={`flex-1 py-4 px-6 rounded-xl font-semibold text-white transition-all duration-200 ${
                isListening || isLoadingVoices
                  ? 'bg-purple-700 cursor-not-allowed opacity-50 shadow-inner'
                  : 'bg-gradient-to-r from-blue-500 via-purple-600 to-cyan-600 hover:from-blue-600 hover:via-purple-700 hover:to-cyan-700 shadow-lg hover:shadow-xl active:scale-95'
              }`}
              type="button"
            >
              {isListening ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></span>
                  Recording...
                </span>
              ) : (
                'Start Recording'
              )}
            </button>
            
            <button
              onClick={stopListening}
              disabled={!isListening}
              className={`flex-1 py-4 px-6 rounded-xl font-semibold text-white transition-all duration-200 ${
                !isListening
                  ? 'bg-purple-700 cursor-not-allowed opacity-50 shadow-inner'
                  : 'bg-gradient-to-r from-purple-500 via-blue-600 to-cyan-600 hover:from-purple-600 hover:via-blue-700 hover:to-cyan-700 shadow-lg hover:shadow-xl active:scale-95'
              }`}
              type="button"
            >
              Stop Recording
            </button>
          </div>

          {/* Audio Player */}
          <div className="bg-purple-900/40 rounded-xl p-4 border border-purple-400/10 shadow-inner">
            <label className="block text-sm font-medium text-blue-300 mb-3">
              Response Audio
            </label>
              <audio 
                ref={audioRef} 
                controls 
                src={audioUrl} 
                className="w-full"
                onPlay={() => setStatus('Playing response...')}
              />
            {!audioUrl && (
              <p className="text-center text-purple-300/50 text-sm mt-2">
                No audio available yet. Start recording to generate audio.
              </p>
            )}
          </div>

        </div>
          


            REAL-TIME CONVERSATION SIDEBAR
{isClient && (
  <>
    <button
  onClick={() => setSidebarOpen(!sidebarOpen)}
  className={`fixed right-0 top-1/2 -translate-y-1/2 z-50 flex items-center transition-all duration-500 rounded-l-3xl shadow-2xl border-2 border-purple-400/60 backdrop-blur-xl ${
    sidebarOpen 
      ? 'bg-gradient-to-l from-purple-600/95 to-pink-600/95 pl-8 pr-4 py-10' 
      : 'bg-gradient-to-l from-purple-600 to-pink-600 pl-4 pr-3 py-12 hover:pl-10 hover:py-14'
  } hover:shadow-purple-500/70`}
>
  {/* Arrow pointing LEFT when closed, RIGHT when open */}
  <svg 
    className={`w-8 h-8 text-white transition-transform duration-500 ${sidebarOpen ? 'rotate-180' : ''}`} 
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
  >
    {/* This path = < arrow */}
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M15 19l-7-7 7-7" />
  </svg>

  {sidebarOpen && (
    <span className="ml-3 text-white font-bold text-lg tracking-wider">Chat</span>
  )}

  {messages.length > 0 && !sidebarOpen && (
    <span className="absolute -left-2 top-4 w-5 h-5 bg-red-500 rounded-full animate-ping border-4 border-black"></span>
  )}
</button>

    {sidebarOpen && (
      <div onClick={() => setSidebarOpen(false)} className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40" />
    )}

    <div
      className={`fixed right-0 top-0 h-screen w-full lg:w-[40vw] bg-gradient-to-b from-purple-950/95 via-blue-950/95 to-black/95 backdrop-blur-3xl border-l border-purple-500/50 shadow-2xl z-50 overflow-y-auto transition-transform duration-500 ${
        sidebarOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
      id="memory-panel"
    >
      <div className="p-8 space-y-6 min-h-full flex flex-col">
        <div className="text-center">
          <h2 className="text-4xl font-extrabold text-white bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            Conversation
          </h2>
        </div>

        <div className="flex-1 space-y-6">
          {messages.length === 0 ? (
            <div className="text-center py-20 text-purple-400/60">
              <div className="text-8xl mb-4">Speaking Head in Silhouette</div>
              <p className="text-xl">Start speaking to begin</p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-start" : "justify-end"} animate-fade-in`}
              >
                <div
                  className={`max-w-xs px-6 py-4 rounded-3xl shadow-xl border backdrop-blur-xl ${
                    msg.role === "user"
                      ? "bg-gradient-to-r from-pink-600/40 to-purple-600/40 border-pink-500/60 text-white"
                      : "bg-gradient-to-r from-cyan-600/40 to-blue-600/40 border-cyan-500/60 text-white"
                  }`}
                >
                  <p className="text-xs font-bold uppercase opacity-80 mb-1">
                    {msg.role === "user" ? "You" : "AI"}
                  </p>
                  <p className="text-lg leading-relaxed">{msg.text}</p>
                </div>
              </div>
            ))
          )}
        </div>

        {messages.length > 0 && (
          <button
            onClick={() => setMessages([])}
            className="w-full py-4 bg-red-600/30 hover:bg-red-600/50 border-2 border-red-500/70 rounded-3xl text-red-300 font-bold"
          >
            Clear Chat
          </button>
        )}
      </div>
    </div>
  </>
)}


        {/* Footer */}
        <div className="text-center mt-6">
          <p className="text-blue-300 text-sm drop-shadow-md">
            Powered by Murf AI Voice Technology
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;