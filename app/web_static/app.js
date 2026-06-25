const stateText = document.getElementById("stateText");
const sourceStream = document.getElementById("sourceStream");
const translatedStream = document.getElementById("translatedStream");
const latencyLine = document.getElementById("latencyLine");
const recordBtn = document.getElementById("recordBtn");
const recordIcon = document.getElementById("recordIcon");
const speakBtn = document.getElementById("speakBtn");
const speakLabel = document.getElementById("speakLabel");
const swapBtn = document.getElementById("swapBtn");
const sourceLang = document.getElementById("sourceLang");
const targetLang = document.getElementById("targetLang");
const textInput = document.getElementById("textInput");
const sendTextBtn = document.getElementById("sendTextBtn");
const sourceHint = document.getElementById("sourceHint");
const targetHint = document.getElementById("targetHint");

const clientSessionId = createClientSessionId();
let debugSequence = 0;
let audioFrameCount = 0;
let audioByteCount = 0;
let lastStateText = "";
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
let pendingSegments = new Set();
let incrementalSegments = new Set();
let partialSourceNodes = new Map();
let stopRequested = false;
let processingWatchdog = null;
let ttsEnabled = false;
let ttsConfigured = false;
let ttsDraining = false;
let ttsGeneration = 0;
let currentSpeech = null;
let currentSpeechStop = null;
let ttsQueue = [];
let ttsFetchChain = Promise.resolve();
let maxPendingTtsChunks = 8;

const PROCESSING_TIMEOUT_MS = 20000;

debugLog("ui_loaded", { href: location.href, user_agent: navigator.userAgent });

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (!ttsConfigured) {
      ttsEnabled = Boolean(data.tts_ui_enabled_by_default);
      maxPendingTtsChunks = Math.max(1, Number(data.tts_max_pending_chunks || 8));
      ttsConfigured = true;
      updateTtsControl();
    }
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

function swapDirection() {
  cancelTtsPlayback();
  direction = direction === "vi2en" ? "en2vi" : "vi2en";
  updateDirection();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", direction }));
  }
  if (ttsEnabled) prepareTtsRuntime();
  latencyLine.textContent = "";
  debugLog("direction_swapped", { direction });
}

swapBtn.addEventListener("click", swapDirection);

document.addEventListener("keydown", (event) => {
  const target = event.target;
  const isTyping = target instanceof HTMLElement
    && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
  if (
    event.key.toLowerCase() !== "q"
    || event.repeat
    || event.ctrlKey
    || event.altKey
    || event.metaKey
    || isTyping
  ) {
    return;
  }
  event.preventDefault();
  swapDirection();
});

recordBtn.addEventListener("click", async () => {
  if (streaming) {
    stopStreaming();
    return;
  }
  try {
    await startStreaming();
  } catch (err) {
    debugLog("start_streaming_failed", { error: String(err.message || err) });
    latencyLine.textContent = String(err.message || err);
    cleanupStreaming(true);
  }
});

async function startStreaming() {
  if (busy || streaming) return;
  debugLog("start_streaming_requested", { direction });

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContext();
  audioFrameCount = 0;
  audioByteCount = 0;
  ws = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/stream?client_session=${encodeURIComponent(clientSessionId)}`,
  );
  ws.binaryType = "arraybuffer";
  ws.onmessage = handleStreamMessage;
  ws.onclose = (event) => {
    debugLog("ws_closed", { code: event.code, reason: event.reason, was_clean: event.wasClean });
    if (streaming) cleanupStreaming(false);
  };
  await waitForSocket(ws);
  debugLog("ws_opened", { direction });
  ws.send(JSON.stringify({ type: "config", direction }));

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(2048, 1, 1);
  muteNode = audioContext.createGain();
  muteNode.gain.value = 0;
  processorNode.onaudioprocess = (event) => {
    if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > 750000) {
      stateText.textContent = "Catching up";
      debugLog("ws_backpressure", { buffered_amount: ws.bufferedAmount });
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const pcm16 = floatToPcm16(downsample(input, audioContext.sampleRate, 16000));
    audioFrameCount += 1;
    audioByteCount += pcm16.byteLength;
    if (audioFrameCount === 1 || audioFrameCount % 50 === 0) {
      debugLog("audio_frame_sent", {
        frames: audioFrameCount,
        bytes: audioByteCount,
        buffered_amount: ws.bufferedAmount,
      });
    }
    ws.send(pcm16.buffer);
  };

  sourceNode.connect(processorNode);
  processorNode.connect(muteNode);
  muteNode.connect(audioContext.destination);

  streaming = true;
  stopRequested = false;
  pendingSegments.clear();
  recordBtn.classList.add("recording");
  recordIcon.textContent = "II";
  latencyLine.textContent = "";
  setUiState("Listening");
}

function stopStreaming() {
  if (!streaming) return;
  debugLog("stop_streaming_requested", {
    pending_segments: pendingSegments.size,
    frames: audioFrameCount,
    bytes: audioByteCount,
  });
  streaming = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    stopRequested = true;
    ws.send(JSON.stringify({ type: "stop" }));
    scheduleDeferredSocketClose();
  }

  cleanupAudioGraph();
  recordBtn.classList.remove("recording");
  recordIcon.textContent = "Mic";
  updateStreamState();
}

function cleanupStreaming(closeSocket = true) {
  debugLog("cleanup_streaming", { close_socket: closeSocket, pending_segments: pendingSegments.size });
  streaming = false;
  cleanupAudioGraph();
  pendingSegments.clear();
  stopRequested = false;
  clearProcessingWatchdog();
  if (closeSocket && ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
  ws = null;
  recordBtn.classList.remove("recording");
  recordIcon.textContent = "Mic";
  updateStreamState();
}

function cleanupAudioGraph() {
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
}

function handleStreamMessage(event) {
  try {
    const message = JSON.parse(event.data);
    debugLog(
      `ws_message_${message.type}${message.status ? `_${message.status}` : ""}`,
      {
        direction: message.direction,
        has_data: Boolean(message.data),
        pending_segments: pendingSegments.size,
      },
      message.segment_id,
    );
    if (message.type === "partial_asr") {
      renderPartialAsr(message);
    } else if (message.type === "commit") {
      renderCommittedSource(message);
    } else if (message.type === "translation_partial") {
      renderCommittedTranslation(message);
    } else if (message.type === "result" && hasRenderableResult(message.data)) {
      renderResult(message.data, message.segment_id);
      markSegmentDone(message.segment_id);
    } else if (message.type === "segment" && message.status === "processing") {
      markSegmentProcessing(message.segment_id);
    } else if (message.type === "segment" && message.status === "accepted") {
      if (message.segment_id != null) pendingSegments.add(message.segment_id);
      updateStreamState();
    } else if (message.type === "segment" && message.status === "done") {
      markSegmentDone(message.segment_id);
    } else if (message.type === "segment" && message.status === "dropped") {
      markSegmentDone(message.segment_id);
      latencyLine.textContent = message.detail || "Dropped stale segment under load";
    } else if (message.type === "segment" && message.status === "ignored") {
      markSegmentDone(message.segment_id);
      latencyLine.textContent = message.detail || "Ignored segment";
    } else if (message.type === "error") {
      markSegmentDone(message.segment_id);
      latencyLine.textContent = message.detail || "Stream error";
    } else if (message.type === "ready") {
      updateStreamState();
    }
  } catch (err) {
    debugLog("ws_message_parse_error", { error: String(err.message || err), raw: String(event.data).slice(0, 500) });
    latencyLine.textContent = String(err.message || err);
  }
}

function hasRenderableResult(data) {
  return Boolean(data && (data.source_text || data.translated_text));
}

function markSegmentProcessing(segmentId) {
  if (segmentId != null) pendingSegments.add(segmentId);
  debugLog("segment_processing_ui", { pending_segments: pendingSegments.size }, segmentId);
  updateStreamState();
  armProcessingWatchdog();
}

function markSegmentDone(segmentId) {
  if (segmentId != null) pendingSegments.delete(segmentId);
  removePartialAsr(segmentId);
  debugLog("segment_done_ui", { pending_segments: pendingSegments.size }, segmentId);
  updateStreamState();
  if (!pendingSegments.size) clearProcessingWatchdog();
  if (stopRequested && !pendingSegments.size) closeSocketSoon();
}

function updateStreamState() {
  if (busy) return;
  if (pendingSegments.size > 0) {
    setUiState(`Translating ${pendingSegments.size} segment${pendingSegments.size > 1 ? "s" : ""}`);
  } else if (streaming) {
    setUiState("Listening");
  } else {
    setUiState("Live translating");
  }
}

function armProcessingWatchdog() {
  clearProcessingWatchdog();
  processingWatchdog = setTimeout(() => {
    if (!pendingSegments.size) return;
    debugLog("processing_watchdog_timeout", { pending_segments: pendingSegments.size });
    pendingSegments.clear();
    latencyLine.textContent = "Segment result timed out in the browser; continuing live stream.";
    updateStreamState();
    if (stopRequested) closeSocketSoon();
  }, PROCESSING_TIMEOUT_MS);
}

function clearProcessingWatchdog() {
  if (processingWatchdog) {
    clearTimeout(processingWatchdog);
    processingWatchdog = null;
  }
}

function scheduleDeferredSocketClose() {
  setTimeout(() => {
    if (!stopRequested || pendingSegments.size > 0) return;
    closeSocketSoon();
  }, 3000);
  setTimeout(() => {
    if (stopRequested) closeSocketSoon();
  }, PROCESSING_TIMEOUT_MS);
}

function closeSocketSoon() {
  debugLog("close_socket", { ready_state: ws ? ws.readyState : null, pending_segments: pendingSegments.size });
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
  ws = null;
  stopRequested = false;
  updateStreamState();
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

speakBtn.addEventListener("click", () => {
  ttsEnabled = !ttsEnabled;
  if (!ttsEnabled) {
    cancelTtsPlayback();
  } else if (lastResult && lastResult.translated_text) {
    prepareTtsRuntime();
    enqueueTts(lastResult);
  } else if (ttsEnabled) {
    prepareTtsRuntime();
  }
  updateTtsControl();
  debugLog("tts_toggled", { enabled: ttsEnabled });
});

async function prepareTtsRuntime() {
  const startedAt = performance.now();
  try {
    const res = await fetch("/api/tts/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction, session_id: clientSessionId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    debugLog("tts_prepare_done", {
      latency_ms: Math.round(performance.now() - startedAt),
      backend: data.runtime && data.runtime.english_backend,
      active_language: data.runtime && data.runtime.active_output_language,
    });
  } catch (err) {
    debugLog("tts_prepare_failed", { error: String(err.message || err) });
  }
}

function renderResult(data, segmentId = null) {
  lastResult = data;
  if (!data.incremental_committed && !incrementalSegments.has(segmentId)) {
    appendTranscriptPair(data);
    enqueueTts(data, segmentId);
  } else {
    removePartialAsr(segmentId);
  }
  debugLog("render_result", {
    source_len: (data.source_text || "").length,
    translated_len: (data.translated_text || "").length,
    latency_ms: data.latency_ms || {},
  }, segmentId);
  const timing = data.latency_ms || {};
  latencyLine.textContent = [
    timing.denoise != null ? `Denoise ${timing.denoise}ms` : null,
    timing.asr != null ? `ASR ${timing.asr}ms` : null,
    timing.mt != null ? `MT ${timing.mt}ms` : null,
    timing.total != null ? `Total ${timing.total}ms` : null,
  ].filter(Boolean).join(" | ");
}

function renderPartialAsr(message) {
  clearPlaceholders();
  const segmentId = message.segment_id;
  let entry = partialSourceNodes.get(segmentId);
  if (!entry) {
    entry = createEntry("", "partial");
    entry.classList.add("partial-entry");
    sourceStream.appendChild(entry);
    partialSourceNodes.set(segmentId, entry);
  }
  entry.querySelector(".entry-text").textContent = message.unstable_text || message.text || "";
  entry.querySelector(".entry-meta").textContent = "partial";
  scheduleScroll();
}

function renderCommittedSource(message) {
  clearPlaceholders();
  incrementalSegments.add(message.segment_id);
  sourceStream.appendChild(createEntry(message.source_text || "", "committed"));
  removePartialAsr(message.segment_id);
  scheduleScroll();
}

function renderCommittedTranslation(message) {
  clearPlaceholders();
  incrementalSegments.add(message.segment_id);
  const timing = message.latency_ms || {};
  const meta = timing.total != null ? `committed | ${timing.total}ms` : "committed";
  translatedStream.appendChild(createEntry(message.translated_text || "", meta));
  lastResult = {
    source_text: message.source_text || "",
    translated_text: message.translated_text || "",
    direction: message.direction || direction,
    emotion: "neutral",
  };
  enqueueTts(lastResult, message.segment_id);
  scheduleScroll();
}

function updateTtsControl() {
  speakBtn.classList.toggle("enabled", ttsEnabled);
  speakBtn.setAttribute("aria-pressed", String(ttsEnabled));
  speakBtn.setAttribute("aria-label", ttsEnabled ? "Silence translated speech" : "Enable translated speech");
  speakLabel.textContent = ttsEnabled ? "Silence" : "Voice";
}

function enqueueTts(result, segmentId = null) {
  const text = String(result && result.translated_text || "").trim();
  if (!ttsEnabled || !text) return;
  const itemDirection = result.direction || direction;
  const duplicate = ttsQueue.some(
    (item) => item.segmentId === segmentId && item.text === text && item.direction === itemDirection,
  );
  if (duplicate) {
    debugLog("tts_deduplicated", { chars: text.length }, segmentId);
    return;
  }

  while (ttsQueue.length >= maxPendingTtsChunks) {
    const stale = ttsQueue.shift();
    if (stale) stale.controller.abort();
  }

  const generation = ttsGeneration;
  const controller = new AbortController();
  const requestId = `${clientSessionId}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const item = {
    controller,
    generation,
    segmentId,
    text,
    direction: itemDirection,
    emotion: result.emotion || "neutral",
    originalText: result.source_text || "",
    requestId,
    requestStarted: false,
  };

  const previous = ttsQueue[ttsQueue.length - 1];
  if (
    previous
    && !previous.requestStarted
    && previous.segmentId === segmentId
    && previous.direction === itemDirection
    && previous.text.length + text.length + 1 <= 80
  ) {
    previous.text = `${previous.text} ${text}`.trim();
    previous.originalText = `${previous.originalText} ${item.originalText}`.trim();
    debugLog("tts_coalesced", { chars: previous.text.length }, segmentId);
    return;
  }

  item.audioPromise = ttsFetchChain.then(async () => {
    if (!ttsEnabled || item.generation !== ttsGeneration) {
      throw new DOMException("TTS request cancelled", "AbortError");
    }
    item.requestStarted = true;
    const startedAt = performance.now();
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: item.text,
        direction: item.direction,
        emotion: item.emotion,
        original_text: item.originalText,
        session_id: clientSessionId,
        segment_id: item.segmentId,
        request_id: item.requestId,
      }),
      signal: item.controller.signal,
    });
    if (!res.ok) throw new Error(await res.text());
    return {
      blob: await res.blob(),
      synthesisMs: Math.round(performance.now() - startedAt),
      inferenceMs: Number(res.headers.get("X-OneVoice-TTS-Inference-Ms") || 0),
      rtf: Number(res.headers.get("X-OneVoice-TTS-RTF") || 0),
      engine: res.headers.get("X-OneVoice-TTS-Engine") || "unknown",
    };
  });
  ttsFetchChain = item.audioPromise.catch(() => {});
  ttsQueue.push(item);
  debugLog("tts_queued", { queue_size: ttsQueue.length, chars: text.length }, segmentId);
  drainTtsQueue();
}

async function drainTtsQueue() {
  if (ttsDraining) return;
  ttsDraining = true;
  try {
    while (ttsEnabled && ttsQueue.length) {
      const item = ttsQueue.shift();
      try {
        const prepared = await item.audioPromise;
        if (!ttsEnabled || item.generation !== ttsGeneration) continue;
        const objectUrl = URL.createObjectURL(prepared.blob);
        const audio = new Audio(objectUrl);
        currentSpeech = audio;
        speakBtn.classList.add("speaking");
        debugLog(
          "tts_play_start",
          {
            synthesis_ms: prepared.synthesisMs,
            inference_ms: prepared.inferenceMs,
            rtf: prepared.rtf,
            engine: prepared.engine,
            remaining: ttsQueue.length,
          },
          item.segmentId,
        );
        try {
          await playAudio(audio);
        } finally {
          URL.revokeObjectURL(objectUrl);
        }
        debugLog("tts_play_done", { remaining: ttsQueue.length }, item.segmentId);
      } catch (err) {
        if (err.name !== "AbortError") {
          latencyLine.textContent = `TTS: ${err.message}`;
          debugLog("tts_failed", { error: String(err.message || err) }, item.segmentId);
        }
      } finally {
        currentSpeech = null;
        currentSpeechStop = null;
        speakBtn.classList.remove("speaking");
      }
    }
  } finally {
    ttsDraining = false;
    if (ttsEnabled && ttsQueue.length) drainTtsQueue();
  }
}

function playAudio(audio) {
  return new Promise((resolve, reject) => {
    const finish = () => resolve();
    currentSpeechStop = finish;
    audio.addEventListener("ended", finish, { once: true });
    audio.addEventListener("error", () => reject(new Error("Browser audio playback failed")), { once: true });
    audio.play().catch(reject);
  });
}

function cancelTtsPlayback() {
  ttsGeneration += 1;
  ttsQueue.forEach((item) => item.controller.abort());
  ttsQueue = [];
  ttsFetchChain = Promise.resolve();
  if (currentSpeech) {
    currentSpeech.pause();
    currentSpeech.currentTime = 0;
    currentSpeech = null;
  }
  if (currentSpeechStop) {
    currentSpeechStop();
    currentSpeechStop = null;
  }
  speakBtn.classList.remove("speaking");
}

function removePartialAsr(segmentId) {
  const entry = partialSourceNodes.get(segmentId);
  if (entry) {
    entry.remove();
    partialSourceNodes.delete(segmentId);
  }
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
    socket.onerror = (event) => {
      debugLog("ws_error", { ready_state: socket.readyState });
      reject(event);
    };
  });
}

function setUiState(value) {
  stateText.textContent = value;
  if (value !== lastStateText) {
    lastStateText = value;
    debugLog("state_change", { state: value, pending_segments: pendingSegments.size });
  }
}

function createClientSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function debugLog(event, details = {}, segmentId = null) {
  debugSequence += 1;
  const payload = {
    session_id: clientSessionId,
    event,
    segment_id: segmentId,
    details: {
      seq: debugSequence,
      streaming,
      stop_requested: stopRequested,
      ws_state: ws ? ws.readyState : null,
      ...details,
    },
  };
  window.oneVoiceDebug = window.oneVoiceDebug || [];
  window.oneVoiceDebug.push({ ts: Date.now(), ...payload });
  if (window.oneVoiceDebug.length > 500) window.oneVoiceDebug.shift();
  fetch("/api/client-log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    keepalive: true,
  }).catch(() => {});
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
updateTtsControl();
ensurePlaceholders();
