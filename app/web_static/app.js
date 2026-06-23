const stateText = document.getElementById("stateText");
const sourceStream = document.getElementById("sourceStream");
const translatedStream = document.getElementById("translatedStream");
const latencyLine = document.getElementById("latencyLine");
const recordBtn = document.getElementById("recordBtn");
const recordIcon = document.getElementById("recordIcon");
const speakBtn = document.getElementById("speakBtn");
const swapBtn = document.getElementById("swapBtn");
const sourceLang = document.getElementById("sourceLang");
const targetLang = document.getElementById("targetLang");
const textInput = document.getElementById("textInput");
const sendTextBtn = document.getElementById("sendTextBtn");
const sourceHint = document.getElementById("sourceHint");
const targetHint = document.getElementById("targetHint");

let direction = "vi2en";
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let muteNode = null;
let ws = null;
let streaming = false;
let lastResult = null;
let busy = false;
let entryIndex = 0;
let scrollPending = false;

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (data.ready) {
      stateText.textContent = "Live translating";
      ensurePlaceholders();
      latencyLine.textContent = "";
      return;
    }
    stateText.textContent = data.error ? "Model error" : "Loading models";
  } catch {
    stateText.textContent = "Connecting";
  }
  setTimeout(pollStatus, 1000);
}

pollStatus();

function updateDirection() {
  if (direction === "vi2en") {
    sourceLang.textContent = "VI";
    targetLang.textContent = "EN";
    sourceHint.textContent = "Vietnamese input";
    targetHint.textContent = "English output";
  } else {
    sourceLang.textContent = "EN";
    targetLang.textContent = "VI";
    sourceHint.textContent = "English input";
    targetHint.textContent = "Vietnamese output";
  }
}

swapBtn.addEventListener("click", () => {
  direction = direction === "vi2en" ? "en2vi" : "vi2en";
  updateDirection();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", direction }));
  }
  latencyLine.textContent = "";
});

recordBtn.addEventListener("click", async () => {
  if (streaming) {
    stopStreaming();
    return;
  }
  await startStreaming();
});

async function startStreaming() {
  if (busy || streaming) return;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContext();
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  await waitForSocket(ws);
  ws.send(JSON.stringify({ type: "config", direction }));
  ws.onmessage = handleStreamMessage;
  ws.onclose = () => {
    if (streaming) stopStreaming();
  };

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(2048, 1, 1);
  muteNode = audioContext.createGain();
  muteNode.gain.value = 0;
  processorNode.onaudioprocess = (event) => {
    if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > 750000) {
      stateText.textContent = "Catching up";
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const pcm16 = floatToPcm16(downsample(input, audioContext.sampleRate, 16000));
    ws.send(pcm16.buffer);
  };

  sourceNode.connect(processorNode);
  processorNode.connect(muteNode);
  muteNode.connect(audioContext.destination);

  streaming = true;
  recordBtn.classList.add("recording");
  recordIcon.textContent = "II";
  latencyLine.textContent = "";
  stateText.textContent = "Listening";
}

function stopStreaming() {
  streaming = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    setTimeout(() => {
      if (ws) ws.close();
    }, 250);
  }

  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
  }
  if (muteNode) muteNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (audioContext) audioContext.close();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());

  processorNode = null;
  sourceNode = null;
  audioContext = null;
  muteNode = null;
  mediaStream = null;
  ws = null;

  recordBtn.classList.remove("recording");
  recordIcon.textContent = "Mic";
  stateText.textContent = "Live translating";
}

function handleStreamMessage(event) {
  try {
    const message = JSON.parse(event.data);
    if (message.type === "result" && message.data?.translated_text) {
      renderResult(message.data);
      stateText.textContent = streaming ? "Listening" : "Live translating";
    } else if (message.type === "segment" && message.status === "processing") {
      stateText.textContent = "Translating segment";
    } else if (message.type === "segment" && message.status === "dropped") {
      latencyLine.textContent = message.detail || "Dropped stale segment under load";
    } else if (message.type === "error") {
      latencyLine.textContent = message.detail || "Stream error";
    } else if (message.type === "ready") {
      stateText.textContent = streaming ? "Listening" : "Live translating";
    }
  } catch (err) {
    latencyLine.textContent = String(err.message || err);
  }
}

sendTextBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text || busy) return;
  busy = true;
  stateText.textContent = "Translating";
  try {
    const res = await fetch("/api/translate-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
    });
    const data = await parseResponse(res);
    renderResult(data);
    textInput.value = "";
  } catch (err) {
    appendSystemEntry("Could not translate.", err.message);
    latencyLine.textContent = "";
  } finally {
    busy = false;
    stateText.textContent = streaming ? "Listening" : "Live translating";
  }
});

textInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendTextBtn.click();
  }
});

speakBtn.addEventListener("click", async () => {
  if (!lastResult || busy) return;
  busy = true;
  speakBtn.classList.add("active");
  stateText.textContent = "Synthesizing";
  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: lastResult.translated_text,
        direction: lastResult.direction,
        emotion: lastResult.emotion || "neutral",
        original_text: lastResult.source_text,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const wav = await res.blob();
    const audio = new Audio(URL.createObjectURL(wav));
    await audio.play();
  } catch (err) {
    latencyLine.textContent = err.message;
  } finally {
    busy = false;
    speakBtn.classList.remove("active");
    stateText.textContent = streaming ? "Listening" : "Live translating";
  }
});

function renderResult(data) {
  lastResult = data;
  appendTranscriptPair(data);
  const timing = data.latency_ms || {};
  latencyLine.textContent = [
    timing.denoise != null ? `Denoise ${timing.denoise}ms` : null,
    timing.asr != null ? `ASR ${timing.asr}ms` : null,
    timing.mt != null ? `MT ${timing.mt}ms` : null,
    timing.total != null ? `Total ${timing.total}ms` : null,
  ].filter(Boolean).join(" | ");
}

function ensurePlaceholders() {
  if (!sourceStream.children.length) {
    sourceStream.innerHTML = '<div class="placeholder">Tap record and start speaking.</div>';
  }
  if (!translatedStream.children.length) {
    translatedStream.innerHTML = '<div class="placeholder">Live translations will appear here.</div>';
  }
}

function clearPlaceholders() {
  sourceStream.querySelectorAll(".placeholder").forEach((node) => node.remove());
  translatedStream.querySelectorAll(".placeholder").forEach((node) => node.remove());
}

function appendTranscriptPair(data) {
  clearPlaceholders();
  entryIndex += 1;
  const timing = data.latency_ms || {};
  const meta = [
    `#${entryIndex}`,
    data.audio_duration_s ? `${data.audio_duration_s}s audio` : null,
    timing.total != null ? `${timing.total}ms` : null,
  ].filter(Boolean).join(" | ");

  sourceStream.appendChild(createEntry(data.source_text || "", meta));
  translatedStream.appendChild(createEntry(data.translated_text || "", meta));
  scheduleScroll();
}

function appendSystemEntry(left, right) {
  clearPlaceholders();
  sourceStream.appendChild(createEntry(left, "system"));
  translatedStream.appendChild(createEntry(right, "system"));
  scheduleScroll();
}

function createEntry(text, meta) {
  const entry = document.createElement("div");
  entry.className = "entry";
  const body = document.createElement("div");
  body.className = "entry-text";
  body.textContent = text;
  const foot = document.createElement("div");
  foot.className = "entry-meta";
  foot.textContent = meta || "";
  entry.append(body, foot);
  return entry;
}

function scheduleScroll() {
  if (scrollPending) return;
  scrollPending = true;
  requestAnimationFrame(() => {
    sourceStream.scrollTop = sourceStream.scrollHeight;
    translatedStream.scrollTop = translatedStream.scrollHeight;
    scrollPending = false;
  });
}

async function parseResponse(res) {
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = data.detail || data.message || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function waitForSocket(socket) {
  return new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });
}

function downsample(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j += 1) sum += input[j];
    output[i] = sum / Math.max(1, end - start);
  }
  return output;
}

function floatToPcm16(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

updateDirection();
ensurePlaceholders();
