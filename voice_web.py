"""
Voice Chat Web UI — WebSocket + Binary Audio
  Browser connects directly to RunPod server via WebSocket
  Binary WAV audio (no base64), streaming TTS playback
  Falls back to HTTP POST if WebSocket unavailable

Usage:
    POD_URL=https://{pod_id}-5300.proxy.runpod.net python3 voice_web.py
    Open http://localhost:5050
"""
import os
import sys

from flask import Flask

# ============================================================
# Config
# ============================================================

POD_URL = os.environ.get("POD_URL", "https://s7p762hm9q1eq8-5300.proxy.runpod.net")
WS_URL = POD_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

app = Flask(__name__)
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return HTML_PAGE.replace("{{WS_URL}}", WS_URL).replace("{{POD_URL}}", POD_URL)


HTML_PAGE = r"""<!DOCTYPE html>
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
.container { max-width: 500px; width: 100%; padding: 20px; }
h1 {
    text-align: center; font-size: 1.5em; margin: 20px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.conn-badge {
    text-align: center; font-size: 0.75em; margin-bottom: 10px;
    padding: 4px 12px; border-radius: 12px; display: inline-block;
}
.conn-badge.ws { background: #1a2e1a; color: #4ade80; }
.conn-badge.http { background: #2e2a1a; color: #fbbf24; }
.conn-badge.disconnected { background: #2e1a1a; color: #f87171; }
.header-row { text-align: center; }
.status {
    text-align: center; padding: 10px; margin: 10px 0;
    border-radius: 10px; font-size: 0.9em; background: #1a1a2e;
}
.status.listening { background: #1a2e1a; color: #4ade80; }
.status.processing { background: #2e2a1a; color: #fbbf24; }
.status.speaking { background: #1a1a2e; color: #818cf8; }
.status.error { background: #2e1a1a; color: #f87171; }

#mic-btn {
    display: block; margin: 30px auto; width: 120px; height: 120px;
    border-radius: 50%; border: 3px solid #333; background: #1a1a2e;
    cursor: pointer; font-size: 40px; transition: all 0.3s;
}
#mic-btn:hover { border-color: #667eea; }
#mic-btn.active {
    border-color: #4ade80; background: #1a2e1a;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }
    50% { box-shadow: 0 0 0 20px rgba(74,222,128,0); }
}

.volume-bar { width: 200px; height: 6px; background: #1a1a2e; border-radius: 3px; margin: 15px auto; overflow: hidden; }
.volume-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #4ade80, #fbbf24, #f87171); transition: width 0.05s; border-radius: 3px; }

.chat-log { margin-top: 20px; max-height: 400px; overflow-y: auto; }
.msg { padding: 10px 14px; margin: 8px 0; border-radius: 12px; font-size: 0.9em; line-height: 1.4; }
.msg.user { background: #1a2e1a; border-left: 3px solid #4ade80; }
.msg.bot { background: #1a1a2e; border-left: 3px solid #818cf8; }
.msg .meta { font-size: 0.75em; color: #666; margin-top: 4px; }
.auto-toggle { text-align: center; margin: 10px 0; }
.auto-toggle label { cursor: pointer; font-size: 0.85em; color: #888; }
.auto-toggle input { margin-right: 5px; }
</style>
</head>
<body>
<div class="container">
    <h1>Voice Chat</h1>
    <div class="header-row">
        <span id="conn-badge" class="conn-badge disconnected">Connecting...</span>
    </div>
    <div class="auto-toggle">
        <label><input type="checkbox" id="auto-mode" checked> Auto-detect speech (VAD)</label>
    </div>
    <div id="status" class="status">Connecting to server...</div>
    <button id="mic-btn">🎤</button>
    <div class="volume-bar"><div class="volume-fill" id="volume"></div></div>
    <div class="chat-log" id="chat-log"></div>
</div>

<script>
// ============================================================
// Config
// ============================================================
const WS_URL = '{{WS_URL}}';
const POD_URL = '{{POD_URL}}';

const SPEECH_THRESHOLD = 25;
const SILENCE_TIMEOUT = 200;   // ms silence before sending (lower = faster response)
const MIN_SPEECH_MS = 250;

// ============================================================
// State
// ============================================================
let ws = null;
let useWebSocket = true;
let mediaStream = null;
let recCtx = null;       // Recording AudioContext (16kHz)
let analyser = null;
let recorder = null;
let chunks = [];
let isRecording = false;
let isPlaying = false;    // Audio is currently playing from speaker
let isAwaitingResponse = false;  // Waiting for server response
let conversation = [];
let silenceTimer = null;
let speechDetected = false;
let speechStart = 0;
let turnTranscripts = {};  // turn_id -> transcript text
let currentTurnId = 0;     // Increment each request — only play matching turn

// Playback
let playCtx = null;      // Playback AudioContext
let nextPlayTime = 0;
let activeSources = [];

const micBtn = document.getElementById('mic-btn');
const statusEl = document.getElementById('status');
const volumeEl = document.getElementById('volume');
const chatLog = document.getElementById('chat-log');
const autoMode = document.getElementById('auto-mode');
const connBadge = document.getElementById('conn-badge');

// ============================================================
// WebSocket
// ============================================================
let wsReconnectTimer = null;

function connectWS() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        useWebSocket = true;
        connBadge.className = 'conn-badge ws';
        connBadge.textContent = 'WebSocket';
        setStatus('listening', 'Connected! Listening...');
        console.log('[WS] Connected');
    };

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            // Binary = [4 bytes turn_id BE][WAV data]
            if (event.data.byteLength < 5) return;  // too small
            const view = new DataView(event.data);
            const serverTurnId = view.getUint32(0, false);  // big-endian
            if (serverTurnId !== currentTurnId) {
                console.log('[WS] Stale audio chunk turn=' + serverTurnId + ' current=' + currentTurnId);
                return;
            }
            const wavData = event.data.slice(4);
            playAudioChunk(wavData, serverTurnId);
        } else {
            // Text = JSON message
            try {
                const msg = JSON.parse(event.data);
                // Filter by turn_id if present
                if (msg.turn_id !== undefined && msg.turn_id !== currentTurnId) {
                    console.log('[WS] Stale message turn=' + msg.turn_id + ' current=' + currentTurnId);
                    return;
                }
                handleServerMessage(msg);
            } catch(e) {
                console.error('[WS] Bad message:', e);
            }
        }
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        if (useWebSocket) {
            connBadge.className = 'conn-badge disconnected';
            connBadge.textContent = 'Reconnecting...';
            wsReconnectTimer = setTimeout(connectWS, 3000);
        }
    };

    ws.onerror = (err) => {
        console.error('[WS] Error, falling back to HTTP');
        useWebSocket = false;
        ws.close();
        connBadge.className = 'conn-badge http';
        connBadge.textContent = 'HTTP (fallback)';
        setStatus('listening', 'Using HTTP fallback. Listening...');
    };
}

function handleServerMessage(msg) {
    switch(msg.type) {
        case 'transcript':
            // Store transcript keyed by turn_id
            turnTranscripts[msg.turn_id || currentTurnId] = msg.text;
            addMsg('user', msg.text);
            setStatus('processing', 'Thinking...');
            break;
        case 'response': {
            const turnId = msg.turn_id || currentTurnId;
            const transcript = turnTranscripts[turnId] || '';
            addMsg('bot', msg.text, msg.latency, msg.metrics);
            if (transcript) {
                conversation.push({role: 'user', content: transcript});
            }
            conversation.push({role: 'assistant', content: msg.text});
            if (conversation.length > 20) conversation = conversation.slice(-20);
            // Clean up old transcripts
            delete turnTranscripts[turnId];
            // Wait for audio to finish, then resume listening
            waitForPlaybackDone();
            break;
        }
        case 'error':
            if (msg.error !== 'Empty transcript') setStatus('error', msg.error);
            isAwaitingResponse = false;
            isPlaying = false;
            if (autoMode.checked) setTimeout(() => setStatus('listening', 'Listening...'), 1000);
            break;
    }
}

function waitForPlaybackDone() {
    const turnAtCall = currentTurnId;
    const checkDone = () => {
        // Stop checking if turn changed (barge-in)
        if (turnAtCall !== currentTurnId) return;
        if (activeSources.length === 0 && (!playCtx || playCtx.currentTime >= nextPlayTime - 0.05)) {
            isAwaitingResponse = false;
            isPlaying = false;
            if (autoMode.checked) setStatus('listening', 'Listening...');
        } else {
            setTimeout(checkDone, 100);
        }
    };
    setTimeout(checkDone, 300);
}

// ============================================================
// Streaming Audio Playback
// ============================================================
function playAudioChunk(wavArrayBuffer, turnId) {
    // Skip if turn has changed (barge-in happened)
    if (turnId !== currentTurnId) return;

    if (!playCtx) {
        playCtx = new AudioContext({ latencyHint: 'interactive' });
    }

    isPlaying = true;
    setStatus('speaking', 'Speaking...');

    // decodeAudioData needs a copy (it detaches the buffer)
    playCtx.decodeAudioData(wavArrayBuffer.slice(0), (audioBuffer) => {
        // Guard AGAIN inside callback — turn may have changed during decode
        if (turnId !== currentTurnId) return;

        const source = playCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playCtx.destination);

        const now = playCtx.currentTime;
        const startTime = Math.max(now + 0.02, nextPlayTime);
        source.start(startTime);
        nextPlayTime = startTime + audioBuffer.duration;

        activeSources.push(source);
        source.onended = () => {
            const idx = activeSources.indexOf(source);
            if (idx >= 0) activeSources.splice(idx, 1);
            if (activeSources.length === 0) isPlaying = false;
        };
    }, (err) => {
        console.error('[Audio] Decode error:', err);
    });
}

function stopPlayback() {
    for (const src of activeSources) {
        try { src.stop(); } catch(e) {}
    }
    activeSources = [];
    nextPlayTime = 0;
}

// ============================================================
// Microphone & VAD
// ============================================================
async function initMic() {
    if (mediaStream) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    recCtx = new AudioContext({ sampleRate: 16000 });
    const source = recCtx.createMediaStreamSource(mediaStream);
    analyser = recCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    if (autoMode.checked) startVAD();
}

function updateVolume() {
    if (!analyser) return 0;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const avg = data.reduce((a, b) => a + b, 0) / data.length;
    volumeEl.style.width = Math.min(100, avg / 128 * 100) + '%';
    return avg;
}

function startVAD() {
    setStatus('listening', 'Listening...');
    function checkVAD() {
        if (!autoMode.checked) {
            requestAnimationFrame(checkVAD);
            return;
        }
        const level = updateVolume();
        if (level > SPEECH_THRESHOLD) {
            if (!isRecording) {
                // ALWAYS stop audio when user speaks — no state check needed
                currentTurnId++;
                stopPlayback();
                isAwaitingResponse = false;
                isPlaying = false;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'interrupt'}));
                }
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
    if (isRecording || isAwaitingResponse) return;
    chunks = [];
    recorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.start(100);
    isRecording = true;
    speechDetected = false;
    micBtn.classList.add('active');
}

function stopRecording() {
    return new Promise(resolve => {
        if (!recorder || recorder.state === 'inactive') { resolve(null); return; }
        recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            const arrayBuf = await blob.arrayBuffer();
            const audioBuf = await recCtx.decodeAudioData(arrayBuf);
            const wav = audioBufferToWav(audioBuf);
            resolve(wav);
        };
        recorder.stop();
        isRecording = false;
        micBtn.classList.remove('active');
    });
}

async function stopAndSend() {
    if (isAwaitingResponse) return;
    const wavBytes = await stopRecording();
    if (!wavBytes || wavBytes.byteLength < 1000) {
        if (autoMode.checked) setStatus('listening', 'Listening...');
        return;
    }

    currentTurnId++;  // New turn — invalidate old audio
    isAwaitingResponse = true;
    isPlaying = false;
    setStatus('processing', 'Processing...');
    nextPlayTime = 0;

    if (useWebSocket && ws && ws.readyState === WebSocket.OPEN) {
        sendViaWebSocket(wavBytes);
    } else {
        sendViaHTTP(wavBytes);
    }
}

// ============================================================
// Send: WebSocket (primary)
// ============================================================
function sendViaWebSocket(wavBytes) {
    // Send binary WAV
    ws.send(wavBytes);
    // Send process command with turn_id so server can tag responses
    ws.send(JSON.stringify({
        type: 'process',
        turn_id: currentTurnId,
        conversation: conversation
    }));
}

// ============================================================
// Send: HTTP POST (fallback)
// ============================================================
async function sendViaHTTP(wavBytes) {
    const b64 = arrayBufferToBase64(wavBytes);

    try {
        const resp = await fetch(POD_URL + '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio_b64: b64, conversation: conversation }),
        });
        const data = await resp.json();

        if (data.error) {
            if (data.error !== 'Empty transcript') setStatus('error', data.error);
            isAwaitingResponse = false; isPlaying = false;
            if (autoMode.checked) setStatus('listening', 'Listening...');
            return;
        }

        addMsg('user', data.transcript);
        addMsg('bot', data.response, data.latency, data.metrics);

        conversation.push({ role: 'user', content: data.transcript });
        conversation.push({ role: 'assistant', content: data.response });
        if (conversation.length > 20) conversation = conversation.slice(-20);

        // Play audio
        if (data.audio_b64) {
            setStatus('speaking', 'Speaking...');
            const wavResp = base64ToArrayBuffer(data.audio_b64);
            playAudioChunk(wavResp, currentTurnId);
            waitForPlaybackDone();
        } else {
            isAwaitingResponse = false; isPlaying = false;
            if (autoMode.checked) setStatus('listening', 'Listening...');
        }
    } catch (e) {
        setStatus('error', 'Connection error: ' + e.message);
        isAwaitingResponse = false; isPlaying = false;
        if (autoMode.checked) setTimeout(() => setStatus('listening', 'Listening...'), 2000);
    }
}

// ============================================================
// UI Helpers
// ============================================================
function setStatus(cls, text) {
    statusEl.className = 'status ' + cls;
    statusEl.textContent = text;
}

function addMsg(role, text, latency, metrics) {
    const div = document.createElement('div');
    div.className = 'msg ' + role;
    let html = (role === 'user' ? '&#127908; ' : '&#129302; ') + escapeHtml(text);
    if (latency) {
        let metaStr = 'STT ' + latency.stt + 's | LLM ' + latency.llm + 's | TTS ' + latency.tts + 's | Total ' + latency.total + 's';
        if (metrics) {
            if (metrics.input_snr != null) metaStr += ' | SNR ' + metrics.input_snr + 'dB';
            if (metrics.stt_confidence != null) metaStr += ' | Conf ' + metrics.stt_confidence;
            if (metrics.noise_suppressed) metaStr += ' | NS &#10003;';
        }
        html += '<div class="meta">' + metaStr + '</div>';
    }
    div.innerHTML = html;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// ============================================================
// Audio encoding helpers
// ============================================================
function audioBufferToWav(buffer) {
    const numCh = 1, sr = buffer.sampleRate, bps = 16;
    const samples = buffer.getChannelData(0);
    const dataLen = samples.length * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const v = new DataView(buf);
    function ws(o, s) { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); }
    ws(0,'RIFF'); v.setUint32(4,36+dataLen,true); ws(8,'WAVE'); ws(12,'fmt ');
    v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,numCh,true);
    v.setUint32(24,sr,true); v.setUint32(28,sr*numCh*bps/8,true);
    v.setUint16(32,numCh*bps/8,true); v.setUint16(34,bps,true);
    ws(36,'data'); v.setUint32(40,dataLen,true);
    let o = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        o += 2;
    }
    return buf;
}

function arrayBufferToBase64(buf) {
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
}

function base64ToArrayBuffer(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
}

// ============================================================
// Event listeners
// ============================================================
micBtn.addEventListener('click', async () => {
    await initMic();
    if (isAwaitingResponse) return;
    if (isRecording) { stopAndSend(); }
    else { startRecording(); setStatus('listening', 'Recording... click to stop'); }
});

autoMode.addEventListener('change', async () => {
    if (autoMode.checked) { await initMic(); startVAD(); }
    else { setStatus('', 'Click mic to record'); }
});

// ============================================================
// Init
// ============================================================
connectWS();
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
    print("Voice Chat Web UI (WebSocket + Binary)")
    print(f"  Pod:  {POD_URL}")
    print(f"  WS:   {WS_URL}")
    print(f"  Open: http://localhost:5050")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5050, debug=False)
