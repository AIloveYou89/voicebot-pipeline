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

POD_URL = os.environ.get("POD_URL", "https://0si46mr0xvqeke-5300.proxy.runpod.net")
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

const SPEECH_THRESHOLD = 30;          // Volume threshold khi idle (listening)
const BARGEIN_THRESHOLD = 45;         // Volume threshold khi đang phát audio (cao hơn để tránh false trigger)
const BARGEIN_SUSTAIN_FRAMES = 8;     // Phải vượt threshold liên tục N frames (~130ms) mới barge-in
const IDLE_SUSTAIN_FRAMES = 3;        // Phải vượt threshold 3 frames liên tục (~50ms) mới bắt đầu record (tránh noise blip)
const SILENCE_TIMEOUT = 300;          // ms silence before sending (tăng từ 150 để tránh gửi clip quá ngắn)
const MIN_SPEECH_MS = 300;
const SPEECH_FREQ_LOW = 300;          // Hz — giọng nói bắt đầu từ ~300Hz
const SPEECH_FREQ_HIGH = 3500;        // Hz — giọng nói chủ yếu dưới 3500Hz

// ============================================================
// State
// ============================================================
let ws = null;
let useWebSocket = true;
let mediaStream = null;
let recCtx = null;       // Recording AudioContext (16kHz)
let analyser = null;
// PCM capture vars initialized in initMic()
let isRecording = false;
let isPlaying = false;    // Audio is currently playing from speaker
let isAwaitingResponse = false;  // Waiting for server response

// Pre-buffer: rolling ring buffer to capture audio BEFORE VAD triggers
const PRE_BUFFER_MS = 300;       // Keep last 300ms of audio
const PRE_BUFFER_SR = 16000;     // Sample rate
const PRE_BUFFER_LEN = Math.ceil(PRE_BUFFER_MS / 1000 * PRE_BUFFER_SR);
let preBuffer = new Float32Array(PRE_BUFFER_LEN);
let preBufferWritePos = 0;
let preBufferFilled = false;     // true once we've wrapped around at least once
let conversation = [];
let silenceTimer = null;
let speechDetected = false;
let speechStart = 0;
let turnTranscripts = {};  // turn_id -> transcript text
let currentTurnId = 0;     // Increment each request — only play matching turn
let speechFrameCount = 0;  // Consecutive frames above threshold (for barge-in)

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
        case 'ignored':
            // Server ignored this audio (noise, empty transcript, low SNR)
            isAwaitingResponse = false;
            if (autoMode.checked) setStatus('listening', 'Listening...');
            break;
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
let pcmSource = null;   // MediaStreamAudioSourceNode
let pcmProcessor = null; // ScriptProcessorNode for PCM capture
let pcmChunks = [];      // Float32Array chunks accumulated during recording

async function initMic() {
    if (mediaStream) return;
    mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    recCtx = new AudioContext({ sampleRate: 16000 });
    pcmSource = recCtx.createMediaStreamSource(mediaStream);
    analyser = recCtx.createAnalyser();
    analyser.fftSize = 512;
    pcmSource.connect(analyser);

    // PCM capture via ScriptProcessorNode (bufferSize=4096 ~256ms at 16kHz)
    pcmProcessor = recCtx.createScriptProcessor(4096, 1, 1);
    pcmProcessor.onaudioprocess = (e) => {
        const samples = e.inputBuffer.getChannelData(0);
        if (isRecording) {
            pcmChunks.push(new Float32Array(samples));
        }
        // Always feed the pre-buffer (ring buffer) so we capture audio before VAD triggers
        for (let i = 0; i < samples.length; i++) {
            preBuffer[preBufferWritePos] = samples[i];
            preBufferWritePos++;
            if (preBufferWritePos >= PRE_BUFFER_LEN) {
                preBufferWritePos = 0;
                preBufferFilled = true;
            }
        }
    };
    pcmSource.connect(pcmProcessor);
    pcmProcessor.connect(recCtx.destination); // must connect to keep processing

    if (autoMode.checked) startVAD();
}

function updateVolume() {
    if (!analyser) return { total: 0, speech: 0, isSpeechLike: false };
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);

    // Total volume
    const total = data.reduce((a, b) => a + b, 0) / data.length;
    volumeEl.style.width = Math.min(100, total / 128 * 100) + '%';

    // Speech band energy (300-3500 Hz)
    const sr = recCtx ? recCtx.sampleRate : 16000;
    const binHz = sr / (analyser.fftSize || 512);
    const lowBin = Math.floor(SPEECH_FREQ_LOW / binHz);
    const highBin = Math.min(Math.ceil(SPEECH_FREQ_HIGH / binHz), data.length - 1);

    let speechSum = 0;
    let speechCount = 0;
    for (let i = lowBin; i <= highBin; i++) {
        speechSum += data[i];
        speechCount++;
    }
    const speech = speechCount > 0 ? speechSum / speechCount : 0;

    // Speech-like = speech band energy is dominant (>60% of total energy)
    const isSpeechLike = total > 10 && (speech / (total + 0.001)) > 0.6;

    return { total, speech, isSpeechLike };
}

function startVAD() {
    setStatus('listening', 'Listening...');
    function checkVAD() {
        if (!autoMode.checked) {
            requestAnimationFrame(checkVAD);
            return;
        }
        const { total, speech, isSpeechLike } = updateVolume();

        // State-based logic:
        // 1. isPlaying = audio playing from speaker → allow barge-in with high threshold
        // 2. isAwaitingResponse && !isPlaying = waiting for server, no audio yet → IGNORE noise
        // 3. idle (neither) = normal listening → start recording with low threshold
        const isIdle = !isPlaying && !isAwaitingResponse && !isRecording;
        const canBargeIn = isPlaying && !isRecording;
        const threshold = canBargeIn ? BARGEIN_THRESHOLD : SPEECH_THRESHOLD;

        // Check if speech-like sound above threshold
        const isSpeech = speech > threshold && isSpeechLike;

        if (isSpeech) {
            if (canBargeIn) {
                // Barge-in mode: require sustained speech frames before interrupting
                speechFrameCount++;
                if (speechFrameCount >= BARGEIN_SUSTAIN_FRAMES) {
                    // Confirmed human speech — interrupt playback
                    currentTurnId++;
                    stopPlayback();
                    isAwaitingResponse = false;
                    isPlaying = false;
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({type: 'interrupt'}));
                    }
                    startRecording();
                    speechStart = Date.now();
                    speechFrameCount = 0;
                }
            } else if (isIdle) {
                // Idle mode: require a few sustained frames to avoid noise blips
                speechFrameCount++;
                if (speechFrameCount >= IDLE_SUSTAIN_FRAMES) {
                    startRecording();
                    speechStart = Date.now();
                    speechFrameCount = 0;
                }
            }
            // If isAwaitingResponse && !isPlaying → do nothing (wait for server)

            if (isRecording) {
                speechDetected = true;
                clearTimeout(silenceTimer);
                silenceTimer = setTimeout(() => {
                    if (isRecording && speechDetected && (Date.now() - speechStart > MIN_SPEECH_MS)) {
                        stopAndSend();
                    }
                }, SILENCE_TIMEOUT);
            }
        } else {
            // Reset sustained counter when no speech detected
            speechFrameCount = 0;
        }
        requestAnimationFrame(checkVAD);
    }
    requestAnimationFrame(checkVAD);
}

function startRecording() {
    if (isRecording || isAwaitingResponse) return;
    // Prepend pre-buffer so we capture the speech onset (first syllables)
    pcmChunks = [];
    if (preBufferFilled) {
        // Ring buffer: read from writePos to end, then 0 to writePos
        const part1 = preBuffer.slice(preBufferWritePos);
        const part2 = preBuffer.slice(0, preBufferWritePos);
        pcmChunks.push(new Float32Array(part1));
        pcmChunks.push(new Float32Array(part2));
    } else if (preBufferWritePos > 0) {
        // Haven't filled the ring buffer yet, just use what we have
        pcmChunks.push(new Float32Array(preBuffer.slice(0, preBufferWritePos)));
    }
    isRecording = true;
    speechDetected = false;
    micBtn.classList.add('active');
}

function stopRecording() {
    return new Promise(resolve => {
        if (!isRecording && pcmChunks.length === 0) { resolve(null); return; }
        isRecording = false;
        micBtn.classList.remove('active');

        // Merge PCM chunks into single Float32Array
        const totalLen = pcmChunks.reduce((acc, c) => acc + c.length, 0);
        if (totalLen === 0) { resolve(null); return; }
        const pcm = new Float32Array(totalLen);
        let offset = 0;
        for (const chunk of pcmChunks) {
            pcm.set(chunk, offset);
            offset += chunk.length;
        }
        pcmChunks = [];

        // Build WAV instantly (just 44-byte header + int16 samples, ~0ms)
        const wav = float32ToWav(pcm, recCtx.sampleRate);
        resolve(wav);
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
        if (latency.first_audio != null) metaStr += ' | 1st audio ' + latency.first_audio + 's';
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
function float32ToWav(samples, sr) {
    // Convert Float32 PCM → WAV with 44-byte header. Instant (~0ms).
    const numCh = 1, bps = 16;
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
