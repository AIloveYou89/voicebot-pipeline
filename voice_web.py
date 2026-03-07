"""
Voice Chat Web UI — Deepgram Streaming STT → OpenAI LLM → SparkTTS on Pod

Optimized: Deepgram WebSocket streaming (real-time STT while speaking)

Usage:
    python3 voice_web.py
    Open http://localhost:5050
"""
import os
import io
import json
import base64
import time
import wave
import asyncio
import threading

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sock import Sock
import requests as http_requests
import websockets
from openai import OpenAI

# ============================================================
# Config
# ============================================================

def load_env():
    env = {}
    env_file = os.path.expanduser("~/.env.agentic")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env[key.strip()] = val.strip()
    return env

env = load_env()

DEEPGRAM_API_KEY = env.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = env.get("OPENAI_API_KEY", "")
TTS_URL = "https://xlw73o6a65o6hz-7860.proxy.runpod.net/tts/b64"

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI nói tiếng Việt. "
    "Trả lời ngắn gọn, tự nhiên, tối đa 2 câu. "
    "CHỈ trả lời bằng tiếng Việt."
)

DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2&language=vi&encoding=linear16&sample_rate=16000&channels=1"
    "&punctuate=true&interim_results=true&vad_events=true&endpointing=300"
)

app = Flask(__name__)
CORS(app)
sock = Sock(app)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
http_session = http_requests.Session()  # Connection pooling

# ============================================================
# Streaming STT via WebSocket
# ============================================================

async def deepgram_stream_stt(audio_chunks_queue, result_holder):
    """Stream audio to Deepgram WebSocket, get real-time transcript."""
    extra_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    try:
        async with websockets.connect(DEEPGRAM_WS_URL, extra_headers=extra_headers) as ws:
            final_transcript = ""

            async def send_audio():
                while True:
                    chunk = await audio_chunks_queue.get()
                    if chunk is None:  # End signal
                        await ws.send(json.dumps({"type": "CloseStream"}))
                        break
                    await ws.send(chunk)

            async def recv_results():
                nonlocal final_transcript
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") == "Results":
                        is_final = data.get("is_final", False)
                        transcript = data["channel"]["alternatives"][0]["transcript"]
                        if is_final and transcript.strip():
                            final_transcript += " " + transcript
                        # Send interim results back
                        result_holder["interim"] = (final_transcript.strip() + " " + transcript).strip() if not is_final else final_transcript.strip()
                        if is_final:
                            result_holder["interim"] = final_transcript.strip()

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_results())

            await send_task
            # Wait a bit for final results after closing
            try:
                await asyncio.wait_for(recv_task, timeout=2.0)
            except asyncio.TimeoutError:
                pass

            result_holder["final"] = final_transcript.strip()
    except Exception as e:
        result_holder["error"] = str(e)


def run_deepgram_stream(audio_chunks, result_holder):
    """Run async Deepgram streaming in a new event loop."""
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()

    async def main():
        # Put all chunks into queue
        q_task = asyncio.create_task(put_chunks(queue, audio_chunks))
        stt_task = asyncio.create_task(deepgram_stream_stt(queue, result_holder))
        await asyncio.gather(q_task, stt_task)

    async def put_chunks(q, chunks_list):
        for chunk in chunks_list:
            await q.put(chunk)
        await q.put(None)  # End signal

    loop.run_until_complete(main())
    loop.close()


# ============================================================
# WebSocket endpoint for real-time voice chat
# ============================================================

@sock.route("/ws/voice")
def voice_ws(ws):
    """WebSocket endpoint: browser streams audio → real-time STT → LLM → TTS."""
    audio_chunks = []
    conversation = []

    while True:
        try:
            msg = ws.receive(timeout=30)
        except Exception:
            break

        if msg is None:
            break

        # Handle text messages (JSON commands)
        if isinstance(msg, str):
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "conversation":
                conversation = data.get("messages", [])
                continue

            if data.get("type") == "end_audio":
                # All audio received, process now
                if not audio_chunks:
                    ws.send(json.dumps({"type": "error", "error": "No audio"}))
                    continue

                total_start = time.time()

                # STT via Deepgram streaming
                t0 = time.time()
                result = {"interim": "", "final": "", "error": None}
                run_deepgram_stream(audio_chunks, result)
                stt_time = time.time() - t0

                transcript = result["final"]
                if result["error"]:
                    print(f"[STT] Error: {result['error']}")
                    ws.send(json.dumps({"type": "error", "error": f"STT: {result['error']}"}))
                    audio_chunks = []
                    continue

                print(f"[STT] {stt_time:.2f}s: {transcript}")

                if not transcript.strip():
                    ws.send(json.dumps({"type": "error", "error": "Empty transcript"}))
                    audio_chunks = []
                    continue

                ws.send(json.dumps({"type": "transcript", "text": transcript}))

                # LLM
                t0 = time.time()
                try:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation + [{"role": "user", "content": transcript}]
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=80,
                        temperature=0.7,
                    )
                    reply = response.choices[0].message.content.strip()
                    llm_time = time.time() - t0
                    print(f"[LLM] {llm_time:.2f}s: {reply}")
                except Exception as e:
                    ws.send(json.dumps({"type": "error", "error": f"LLM: {e}"}))
                    audio_chunks = []
                    continue

                ws.send(json.dumps({"type": "llm_reply", "text": reply}))

                # TTS
                t0 = time.time()
                try:
                    resp = http_session.post(TTS_URL, json={"text": reply}, timeout=30)
                    resp.raise_for_status()
                    tts_data = resp.json()
                    tts_audio_b64 = tts_data.get("audio_b64", "")
                    tts_time = time.time() - t0
                    print(f"[TTS] {tts_time:.2f}s")
                except Exception as e:
                    tts_audio_b64 = ""
                    tts_time = time.time() - t0
                    print(f"[TTS] Error: {e}")

                total = time.time() - total_start
                print(f"[TOTAL] {total:.2f}s")

                ws.send(json.dumps({
                    "type": "result",
                    "transcript": transcript,
                    "response": reply,
                    "audio_b64": tts_audio_b64,
                    "latency": {
                        "stt": round(stt_time, 2),
                        "llm": round(llm_time, 2),
                        "tts": round(tts_time, 2),
                        "total": round(total, 2),
                    }
                }))

                audio_chunks = []
                continue

        # Binary message = audio chunk
        if isinstance(msg, bytes):
            audio_chunks.append(msg)


# ============================================================
# Fallback REST API (keep for compatibility)
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    audio_b64 = data.get("audio_b64", "")
    conversation = data.get("conversation", [])

    if not audio_b64:
        return jsonify({"error": "No audio"}), 400

    audio_bytes = base64.b64decode(audio_b64)
    total_start = time.time()

    # STT (Deepgram REST - fallback)
    try:
        t0 = time.time()
        resp = http_session.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&language=vi",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            data=audio_bytes,
            timeout=10,
        )
        resp.raise_for_status()
        transcript = resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        stt_time = time.time() - t0

        if not transcript.strip():
            return jsonify({"error": "Empty transcript"}), 200
    except Exception as e:
        return jsonify({"error": f"STT failed: {e}"}), 500

    # LLM
    try:
        t0 = time.time()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation + [{"role": "user", "content": transcript}]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=80,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        llm_time = time.time() - t0
    except Exception as e:
        return jsonify({"error": f"LLM failed: {e}"}), 500

    # TTS
    try:
        t0 = time.time()
        resp = http_session.post(TTS_URL, json={"text": reply}, timeout=30)
        resp.raise_for_status()
        tts_data = resp.json()
        tts_audio_b64 = tts_data.get("audio_b64", "")
        tts_time = time.time() - t0
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500

    total = time.time() - total_start
    return jsonify({
        "transcript": transcript,
        "response": reply,
        "audio_b64": tts_audio_b64,
        "latency": {
            "stt": round(stt_time, 2),
            "llm": round(llm_time, 2),
            "tts": round(tts_time, 2),
            "total": round(total, 2),
        }
    })


# ============================================================
# Web UI
# ============================================================

@app.route("/")
def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Chat</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, system-ui, sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.container {
    max-width: 500px;
    width: 100%;
    padding: 20px;
}
h1 {
    text-align: center;
    font-size: 1.5em;
    margin: 20px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.status {
    text-align: center;
    padding: 10px;
    margin: 10px 0;
    border-radius: 10px;
    font-size: 0.9em;
    background: #1a1a2e;
}
.status.listening { background: #1a2e1a; color: #4ade80; }
.status.processing { background: #2e2a1a; color: #fbbf24; }
.status.speaking { background: #1a1a2e; color: #818cf8; }
.status.error { background: #2e1a1a; color: #f87171; }
.status.transcribing { background: #1a2e2e; color: #22d3ee; }

#mic-btn {
    display: block;
    margin: 30px auto;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 3px solid #333;
    background: #1a1a2e;
    cursor: pointer;
    font-size: 40px;
    transition: all 0.3s;
}
#mic-btn:hover { border-color: #667eea; }
#mic-btn.active {
    border-color: #4ade80;
    background: #1a2e1a;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }
    50% { box-shadow: 0 0 0 20px rgba(74,222,128,0); }
}

.volume-bar {
    width: 200px;
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
    margin: 15px auto;
    overflow: hidden;
}
.volume-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #4ade80, #fbbf24, #f87171);
    transition: width 0.05s;
    border-radius: 3px;
}

.chat-log {
    margin-top: 20px;
    max-height: 400px;
    overflow-y: auto;
}
.msg {
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 12px;
    font-size: 0.9em;
    line-height: 1.4;
}
.msg.user { background: #1a2e1a; border-left: 3px solid #4ade80; }
.msg.bot { background: #1a1a2e; border-left: 3px solid #818cf8; }
.msg .meta { font-size: 0.75em; color: #666; margin-top: 4px; }
.auto-toggle {
    text-align: center;
    margin: 10px 0;
}
.auto-toggle label { cursor: pointer; font-size: 0.85em; color: #888; }
.auto-toggle input { margin-right: 5px; }
</style>
</head>
<body>
<div class="container">
    <h1>Voice Chat</h1>
    <div class="auto-toggle">
        <label><input type="checkbox" id="auto-mode" checked> Auto-detect speech (VAD)</label>
    </div>
    <div id="status" class="status">Click mic or enable auto-detect</div>
    <button id="mic-btn">🎤</button>
    <div class="volume-bar"><div class="volume-fill" id="volume"></div></div>
    <div class="chat-log" id="chat-log"></div>
</div>

<script>
let mediaStream = null;
let audioContext = null;
let analyser = null;
let isRecording = false;
let isProcessing = false;
let conversation = [];
let silenceTimer = null;
let speechDetected = false;
let ws = null;
let scriptProcessor = null;

const micBtn = document.getElementById('mic-btn');
const statusEl = document.getElementById('status');
const volumeEl = document.getElementById('volume');
const chatLog = document.getElementById('chat-log');
const autoMode = document.getElementById('auto-mode');

// VAD params
const SPEECH_THRESHOLD = 25;
const SILENCE_TIMEOUT = 700;
const MIN_SPEECH_MS = 300;
let speechStart = 0;

// Connect WebSocket
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws/voice');
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => console.log('[WS] Connected');
    ws.onclose = () => {
        console.log('[WS] Disconnected, reconnecting...');
        setTimeout(connectWS, 1000);
    };
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        handleWSMessage(data);
    };
}

function handleWSMessage(data) {
    if (data.type === 'transcript') {
        addMsg('user', data.text);
        setStatus('processing', 'Thinking...');
    } else if (data.type === 'llm_reply') {
        // Show reply immediately, before TTS
        setStatus('speaking', 'Generating voice...');
    } else if (data.type === 'result') {
        // Update bot message with full result
        addMsg('bot', data.response, data.latency);
        conversation.push({ role: 'user', content: data.transcript });
        conversation.push({ role: 'assistant', content: data.response });
        if (conversation.length > 20) conversation = conversation.slice(-20);

        if (data.audio_b64) {
            setStatus('speaking', 'Speaking...');
            playAudio(data.audio_b64).then(() => {
                isProcessing = false;
                if (autoMode.checked) setStatus('listening', 'Listening...');
            });
        } else {
            isProcessing = false;
            if (autoMode.checked) setStatus('listening', 'Listening...');
        }
    } else if (data.type === 'error') {
        if (data.error !== 'Empty transcript') {
            setStatus('error', data.error);
        }
        isProcessing = false;
        if (autoMode.checked) setStatus('listening', 'Listening...');
    }
}

// Init mic with ScriptProcessor for raw PCM
async function initMic() {
    if (mediaStream) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    // ScriptProcessor to capture raw PCM for streaming
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
    scriptProcessor.onaudioprocess = (e) => {
        if (!isRecording) return;
        const float32 = e.inputBuffer.getChannelData(0);
        // Convert float32 to int16
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        // Send raw PCM via WebSocket
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(int16.buffer);
        }
    };

    connectWS();
    if (autoMode.checked) startVAD();
}

// Volume meter
function updateVolume() {
    if (!analyser) return 0;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const avg = data.reduce((a, b) => a + b, 0) / data.length;
    volumeEl.style.width = Math.min(100, avg / 128 * 100) + '%';
    return avg;
}

// VAD
function startVAD() {
    setStatus('listening', 'Listening...');
    function checkVAD() {
        if (!autoMode.checked || isProcessing) {
            requestAnimationFrame(checkVAD);
            return;
        }
        const level = updateVolume();
        if (level > SPEECH_THRESHOLD) {
            if (!isRecording) {
                startRecording();
                speechStart = Date.now();
            }
            speechDetected = true;
            clearTimeout(silenceTimer);
            silenceTimer = setTimeout(() => {
                if (isRecording && speechDetected && (Date.now() - speechStart > MIN_SPEECH_MS)) {
                    stopAndSend();
                }
            }, SILENCE_TIMEOUT);
        }
        requestAnimationFrame(checkVAD);
    }
    requestAnimationFrame(checkVAD);
}

function startRecording() {
    if (isRecording || isProcessing) return;
    isRecording = true;
    speechDetected = false;
    micBtn.classList.add('active');
    // Send conversation context
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'conversation', messages: conversation }));
    }
}

function stopAndSend() {
    if (isProcessing || !isRecording) return;
    isRecording = false;
    isProcessing = true;
    micBtn.classList.remove('active');
    setStatus('processing', 'Processing speech...');
    // Signal end of audio
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'end_audio' }));
    }
}

// Play audio
function playAudio(b64) {
    return new Promise(resolve => {
        const audio = new Audio('data:audio/wav;base64,' + b64);
        audio.onended = resolve;
        audio.onerror = resolve;
        audio.play();
    });
}

// UI
function setStatus(cls, text) {
    statusEl.className = 'status ' + cls;
    statusEl.textContent = text;
}

function addMsg(role, text, latency) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    let html = (role === 'user' ? '🎤 ' : '🤖 ') + text;
    if (latency) {
        html += '<div class="meta">STT ' + latency.stt + 's | LLM ' + latency.llm + 's | TTS ' + latency.tts + 's | Total ' + latency.total + 's</div>';
    }
    div.innerHTML = html;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// Manual mic
micBtn.addEventListener('click', async () => {
    await initMic();
    if (isProcessing) return;
    if (isRecording) {
        stopAndSend();
    } else {
        startRecording();
        setStatus('listening', 'Recording... click to stop');
    }
});

// Auto mode toggle
autoMode.addEventListener('change', async () => {
    if (autoMode.checked) {
        await initMic();
        startVAD();
    } else {
        setStatus('', 'Click mic to record');
    }
});

// Auto-init
initMic().catch(() => setStatus('', 'Click mic to start'));
</script>
</body>
</html>
"""

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Voice Chat Web UI (Streaming STT)")
    print(f"  STT: Deepgram Nova-2 (WebSocket streaming)")
    print(f"  LLM: OpenAI gpt-4o-mini (max_tokens=80)")
    print(f"  TTS: SparkTTS on RunPod Pod")
    print(f"  TTS URL: {TTS_URL}")
    print(f"  Open: http://localhost:5050")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5050, debug=False)
