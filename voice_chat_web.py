"""
Real-time Voice Chat — Web UI.

Runs a local web server with:
- Browser-based mic recording (auto-detect speech)
- Proxy to RunPod API (handles CORS)
- Auto-play audio response
- Continuous conversation loop

Usage:
    pip install flask flask-cors requests
    python voice_chat_web.py

Then open http://localhost:5050 in browser.
"""
import os
import sys
import json
import time
import base64

from flask import Flask, request, jsonify, send_from_directory, Response
import requests as http_requests

# ============================================================
# Config
# ============================================================
RUNPOD_API_KEY = ""
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "5wffpipznb10s8")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

app = Flask(__name__)


def load_api_key():
    global RUNPOD_API_KEY
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
    if RUNPOD_API_KEY:
        return

    env_file = os.path.expanduser("~/.env.agentic")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY="):
                    RUNPOD_API_KEY = line.strip().split("=", 1)[1]
                    return

    print("ERROR: RUNPOD_API_KEY not found")
    sys.exit(1)


# ============================================================
# API Proxy
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """Proxy: receive audio, call RunPod, return streaming results."""
    data = request.json
    audio_b64 = data.get("audio_b64")
    conversation = data.get("conversation", [])
    system_prompt = data.get("system_prompt")

    if not audio_b64:
        return jsonify({"error": "No audio"}), 400

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    # Submit job
    payload = {
        "input": {
            "audio_b64": audio_b64,
            "stream": True,
            "conversation": conversation[-6:],
            "system_prompt": system_prompt,
        }
    }

    try:
        resp = http_requests.post(f"{RUNPOD_URL}/run", json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        job_id = resp.json()["id"]
    except Exception as e:
        return jsonify({"error": f"Submit failed: {e}"}), 500

    # Poll for results (max 120s for cold start)
    start = time.time()
    while time.time() - start < 120:
        try:
            resp = http_requests.get(f"{RUNPOD_URL}/stream/{job_id}", headers=headers, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            if result.get("stream"):
                # Extract chunks
                chunks = []
                for chunk in result["stream"]:
                    output = chunk.get("output", chunk)
                    if isinstance(output, dict):
                        chunks.append(output)
                return jsonify({"status": "ok", "chunks": chunks})

            status = result.get("status", "UNKNOWN")
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                return jsonify({"error": f"Job {status}"}), 500

        except Exception as e:
            pass

        time.sleep(0.5)

    return jsonify({"error": "Timeout"}), 504


# ============================================================
# Serve HTML
# ============================================================

@app.route("/")
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VoiceBot — Real-time Chat</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Be Vietnam Pro', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .container {
    max-width: 480px;
    width: 100%;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
  }

  h1 {
    font-size: 1.4rem;
    font-weight: 600;
    margin: 30px 0 5px;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .subtitle {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 30px;
  }

  /* Mic button */
  .mic-area {
    position: relative;
    margin: 20px 0 30px;
  }

  .mic-btn {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 3px solid #222;
    background: radial-gradient(circle at 40% 40%, #1a1a2e, #0a0a0f);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    position: relative;
    z-index: 2;
  }

  .mic-btn:hover { border-color: #00d4ff44; }

  .mic-btn.listening {
    border-color: #00d4ff;
    box-shadow: 0 0 30px #00d4ff33;
    animation: pulse-listen 2s ease-in-out infinite;
  }

  .mic-btn.recording {
    border-color: #ff4444;
    box-shadow: 0 0 40px #ff444444;
    animation: pulse-record 0.8s ease-in-out infinite;
  }

  .mic-btn.processing {
    border-color: #ffa500;
    box-shadow: 0 0 30px #ffa50033;
    animation: spin-border 1.5s linear infinite;
  }

  .mic-btn.playing {
    border-color: #00ff88;
    box-shadow: 0 0 30px #00ff8833;
    animation: pulse-play 1s ease-in-out infinite;
  }

  .mic-btn svg { width: 40px; height: 40px; fill: #888; transition: fill 0.3s; }
  .mic-btn.listening svg { fill: #00d4ff; }
  .mic-btn.recording svg { fill: #ff4444; }
  .mic-btn.processing svg { fill: #ffa500; }
  .mic-btn.playing svg { fill: #00ff88; }

  @keyframes pulse-listen {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.03); }
  }
  @keyframes pulse-record {
    0%, 100% { transform: scale(1); box-shadow: 0 0 30px #ff444433; }
    50% { transform: scale(1.05); box-shadow: 0 0 50px #ff444466; }
  }
  @keyframes pulse-play {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
  }
  @keyframes spin-border {
    0% { border-color: #ffa500 transparent #ffa500 transparent; }
    50% { border-color: transparent #ffa500 transparent #ffa500; }
    100% { border-color: #ffa500 transparent #ffa500 transparent; }
  }

  /* Status */
  .status {
    font-size: 0.9rem;
    font-weight: 500;
    margin: 10px 0;
    min-height: 24px;
    text-align: center;
  }

  .status.idle { color: #666; }
  .status.listening { color: #00d4ff; }
  .status.recording { color: #ff4444; }
  .status.processing { color: #ffa500; }
  .status.playing { color: #00ff88; }

  /* Chat log */
  .chat-log {
    width: 100%;
    flex: 1;
    overflow-y: auto;
    margin-top: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .msg {
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 90%;
    font-size: 0.9rem;
    line-height: 1.5;
  }

  .msg.user {
    align-self: flex-end;
    background: #1a1a3e;
    border: 1px solid #2a2a5e;
    color: #c0c0ff;
  }

  .msg.bot {
    align-self: flex-start;
    background: #0d2818;
    border: 1px solid #1a4a2a;
    color: #a0ffa0;
  }

  .msg .meta {
    font-size: 0.7rem;
    color: #555;
    margin-top: 4px;
  }

  /* Volume visualizer */
  .visualizer {
    display: flex;
    gap: 3px;
    align-items: center;
    height: 30px;
    margin: 5px 0;
  }
  .visualizer .bar {
    width: 4px;
    background: #00d4ff;
    border-radius: 2px;
    transition: height 0.05s ease;
    min-height: 3px;
  }

  .controls {
    margin-top: 20px;
    display: flex;
    gap: 10px;
  }
  .controls button {
    padding: 8px 20px;
    border-radius: 20px;
    border: 1px solid #333;
    background: #111;
    color: #aaa;
    font-family: inherit;
    cursor: pointer;
    font-size: 0.8rem;
  }
  .controls button:hover { border-color: #00d4ff; color: #00d4ff; }
</style>
</head>
<body>

<div class="container">
  <h1>VoiceBot</h1>
  <p class="subtitle">Nhấn nút mic rồi nói chuyện</p>

  <div class="mic-area">
    <button class="mic-btn idle" id="micBtn" onclick="toggleVoiceChat()">
      <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
    </button>
  </div>

  <div class="visualizer" id="visualizer" style="display:none;">
    <div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div><div class="bar" style="height:3px"></div>
  </div>

  <div class="status idle" id="status">Nhấn mic để bắt đầu</div>

  <div class="chat-log" id="chatLog"></div>

  <div class="controls">
    <button onclick="clearChat()">Xóa hội thoại</button>
  </div>
</div>

<script>
const micBtn = document.getElementById('micBtn');
const statusEl = document.getElementById('status');
const chatLog = document.getElementById('chatLog');
const visualizer = document.getElementById('visualizer');
const bars = visualizer.querySelectorAll('.bar');

let isActive = false;
let mediaRecorder = null;
let audioContext = null;
let analyser = null;
let stream = null;
let chunks = [];
let conversation = [];
let silenceTimer = null;
let isRecording = false;
let animFrame = null;

const SILENCE_THRESHOLD = 0.04;  // Higher = less sensitive to background noise
const SILENCE_DURATION = 1500; // ms
const MIN_RECORDING_MS = 800;  // Minimum recording to avoid noise triggers
let recordingStart = 0;

function setState(state, text) {
  micBtn.className = 'mic-btn ' + state;
  statusEl.className = 'status ' + state;
  statusEl.textContent = text;
}

async function toggleVoiceChat() {
  if (isActive) {
    stopAll();
    return;
  }

  isActive = true;
  await startListening();
}

async function startListening() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    setState('listening', 'Đang nghe... nói gì đi!');
    visualizer.style.display = 'flex';

    monitorAudio();
  } catch (e) {
    setState('idle', 'Không thể truy cập mic: ' + e.message);
    isActive = false;
  }
}

function monitorAudio() {
  const dataArray = new Float32Array(analyser.fftSize);

  function check() {
    if (!isActive) return;

    analyser.getFloatTimeDomainData(dataArray);
    let rms = 0;
    for (let i = 0; i < dataArray.length; i++) rms += dataArray[i] * dataArray[i];
    rms = Math.sqrt(rms / dataArray.length);

    // Update visualizer
    updateVisualizer(rms);

    if (rms > SILENCE_THRESHOLD) {
      if (!isRecording) startRecording();
      clearTimeout(silenceTimer);
      silenceTimer = setTimeout(() => {
        if (isRecording) stopRecording();
      }, SILENCE_DURATION);
    }

    animFrame = requestAnimationFrame(check);
  }
  check();
}

function updateVisualizer(rms) {
  const scale = Math.min(rms * 50, 1);
  bars.forEach((bar, i) => {
    const h = 3 + scale * (15 + Math.sin(Date.now() / 100 + i) * 10);
    bar.style.height = h + 'px';
    bar.style.background = isRecording ? '#ff4444' : '#00d4ff';
  });
}

function startRecording() {
  isRecording = true;
  recordingStart = Date.now();
  chunks = [];

  mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = () => processRecording();
  mediaRecorder.start(100);

  setState('recording', '🔴 Đang ghi âm...');
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

  const duration = Date.now() - recordingStart;
  if (duration < MIN_RECORDING_MS) {
    // Too short, ignore
    mediaRecorder.stop();
    chunks = [];
    setState('listening', 'Đang nghe... nói gì đi!');
    isRecording = false;
    return;
  }

  isRecording = false;
  mediaRecorder.stop();
  setState('processing', '⏳ Đang xử lý...');
}

async function processRecording() {
  if (chunks.length === 0) {
    setState('listening', 'Đang nghe... nói gì đi!');
    return;
  }

  const blob = new Blob(chunks, { type: 'audio/webm' });

  // Convert webm to wav using AudioContext
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const wavB64 = audioBufferToWavB64(audioBuffer);

  // Send to API
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_b64: wavB64,
        conversation: conversation.slice(-6),
      })
    });

    const data = await resp.json();
    if (data.error) {
      addMessage('bot', '❌ ' + data.error);
      setState('listening', 'Đang nghe... nói gì đi!');
      return;
    }

    // Process chunks
    let transcript = '';
    let fullResponse = '';
    let latency = {};
    const audioChunks = [];

    console.log('Received chunks:', JSON.stringify(data.chunks?.map(c => ({type: c.type, text: c.text?.substring(0, 50)}))));

    for (const chunk of (data.chunks || [])) {
      if (chunk.type === 'transcript') {
        transcript = chunk.text;
        addMessage('user', transcript);
      } else if (chunk.type === 'audio_chunk') {
        audioChunks.push(chunk);
      } else if (chunk.type === 'done') {
        fullResponse = chunk.response;
        latency = chunk.latency || {};
      } else if (chunk.type === 'error') {
        addMessage('bot', '❌ Lỗi: ' + (chunk.error_code || 'unknown'));
      }
    }

    // Show bot response text immediately, then play audio
    if (fullResponse) {
      const meta = 'STT=' + (latency.stt||'?') + 's | LLM=' + (latency.llm_first_phrase||'?') + 's | TTS=' + (latency.tts_total||'?') + 's | Total=' + (latency.total||'?') + 's';
      addMessage('bot', fullResponse, meta);
      conversation.push({ role: 'user', content: transcript });
      conversation.push({ role: 'assistant', content: fullResponse });
    }

    // Play audio chunks sequentially
    if (audioChunks.length > 0) {
      setState('playing', '🔊 Bot đang nói...');
      cancelAnimationFrame(animFrame);

      for (const chunk of audioChunks) {
        if (chunk.audio_b64) {
          console.log('Playing chunk:', chunk.text?.substring(0, 50));
          try {
            await playAudioB64(chunk.audio_b64);
          } catch(e) {
            console.error('Audio play error:', e);
          }
        }
      }
    } else if (!fullResponse && !transcript) {
      console.log('No response data - raw:', JSON.stringify(data));
    }

  } catch (e) {
    addMessage('bot', '❌ Lỗi kết nối: ' + e.message);
  }

  // Back to listening — resume mic monitoring
  if (isActive) {
    setState('listening', 'Đang nghe... nói gì đi!');
    monitorAudio();
  }
}

function playAudioB64(b64) {
  return new Promise((resolve) => {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    audioContext.decodeAudioData(bytes.buffer.slice(0), (buffer) => {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      source.onended = resolve;
      source.start();
    }, () => resolve());
  });
}

function audioBufferToWavB64(buffer) {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const samples = buffer.getChannelData(0);
  const int16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
  }

  const wavBuffer = new ArrayBuffer(44 + int16.length * 2);
  const view = new DataView(wavBuffer);

  // WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + int16.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, int16.length * 2, true);

  const dataView = new Int16Array(wavBuffer, 44);
  dataView.set(int16);

  // Base64
  const bytes = new Uint8Array(wavBuffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function addMessage(role, text, meta) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = text + (meta ? '<div class="meta">' + meta + '</div>' : '');
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function clearChat() {
  chatLog.innerHTML = '';
  conversation = [];
}

function stopAll() {
  isActive = false;
  isRecording = false;
  cancelAnimationFrame(animFrame);
  clearTimeout(silenceTimer);
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (audioContext) audioContext.close();
  visualizer.style.display = 'none';
  setState('idle', 'Nhấn mic để bắt đầu');
}
</script>

</body>
</html>
"""

if __name__ == "__main__":
    load_api_key()
    print("=" * 50)
    print("  🤖 VoiceBot — Real-time Voice Chat")
    print("=" * 50)
    print(f"  Open: http://localhost:5050")
    print(f"  RunPod endpoint: {ENDPOINT_ID}")
    print("  Ctrl+C to quit\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
