# OneVoice Edge

## Solution and Engineering Overview

**Document status:** Proof-of-concept engineering baseline  
**Primary use case:** Real-time Vietnamese and English voice translation  
**Deployment goal:** Fully local operation on laptops and edge devices  
**Audience:** Competition judges, product teams, business stakeholders, AI engineers, and system engineers

---

## 1. Executive Summary

OneVoice Edge is an offline bilingual speech translation system designed for
natural Vietnamese and English conversations in environments where internet
access is limited, unreliable, or prohibited.

The solution listens continuously, displays speech while the user is still
talking, commits only stable portions of the recognized text, translates those
stable portions, and optionally speaks the translation aloud. This avoids the
two common extremes of voice translation:

- Waiting for the entire sentence, which creates noticeable delay.
- Translating every unstable word, which produces flickering text, repeated
  translations, and unnatural speech.

OneVoice instead uses a two-layer streaming approach:

1. **Partial speech text** is shown immediately and may still change.
2. **Stable committed text** is translated and optionally synthesized as speech.

The system currently supports:

- Vietnamese speech to English text and speech.
- English speech to Vietnamese text and speech.
- Local model execution without an internet dependency during use.
- Optional male speech output in both languages.
- Full-screen retained transcripts for live operational use.
- Runtime measurements and combined browser/backend diagnostics.

---

## 2. Problem Definition

Traditional speech translation systems are commonly optimized for cloud
environments. They assume reliable connectivity, large servers, and permission
to transmit user audio outside the device.

The OneVoice challenge requires a different engineering approach:

- Models must run locally.
- The interaction must feel conversational.
- The system must operate faster than the audio duration.
- Translation must begin quickly after useful speech becomes available.
- Speech output must be understandable and natural.
- The hardware must remain practical and sustainable.

The main engineering problem is therefore not simply translation accuracy. It
is the coordination of audio capture, speech detection, recognition,
translation, rendering, and synthesis under a strict latency and memory budget.

---

## 3. Product Experience

The primary interface is a full-screen bilingual workspace:

- The left side retains the recognized source-language speech.
- The right side retains the translated text.
- Unstable recognition appears as temporary partial text.
- Stable text and translations remain visible as conversation history.
- Both panels scroll automatically as the conversation grows.
- The language direction can be changed between Vietnamese and English.
- The Voice control enables translated speech.
- The Silence control immediately stops current and queued speech.

Text remains available even when speech output is disabled. This keeps the
lowest-latency visual translation experience independent from the optional TTS
workload.

---

## 4. High-Level Architecture

The system is organized as a sequence of cooperating stages:

1. **Microphone capture**
   - Captures mono audio continuously.
   - Converts browser audio to a consistent 16 kHz format.

2. **Streaming voice activity detection**
   - Detects when speech begins and ends.
   - Retains a small amount of audio before speech to avoid losing first words.
   - Adds limited trailing audio to protect final words.

3. **Audio conditioning**
   - Normalizes audio for downstream recognition.
   - Uses noise-robust acoustic processing where available.

4. **Streaming speech recognition**
   - Produces repeated partial recognition snapshots while speech continues.
   - Uses language-specific local ASR models.

5. **Stable text buffer**
   - Separates temporary recognition from committed recognition.
   - Commits text only when a prefix remains consistent across observations.
   - Holds back the most recent words because they are the most likely to change.

6. **Incremental machine translation**
   - Translates committed source chunks rather than every partial update.
   - Preserves low latency without repeatedly translating unstable words.

7. **Live user interface**
   - Updates partial and committed content independently.
   - Retains the complete conversation history.

8. **Optional text-to-speech**
   - Synthesizes committed translated chunks.
   - Runs outside the critical ASR and translation path.
   - Uses a bounded, ordered queue to prevent speech backlog.

9. **Observability**
   - Records browser events and backend events in a shared timeline.
   - Measures model inference time, queue behavior, playback timing, and RTF.

---

## 5. Core Streaming Strategy

### 5.1 Partial Recognition

While the user is speaking, OneVoice periodically recognizes the active audio
window. This text appears immediately in a visually temporary state.

Partial text is useful for responsiveness, but it is not treated as final. The
ASR model may revise recent words when more acoustic context becomes available.

### 5.2 Stable Commitment

A recognized prefix becomes stable when it remains consistent across repeated
partial observations. The system also protects the newest words by holding them
back temporarily.

This policy reduces:

- Word flicker.
- Duplicate translations.
- Missing or replaced sentence endings.
- TTS speaking text that is immediately corrected.

### 5.3 Finalization

When the VAD detects the end of a speech segment, the remaining uncommitted text
is finalized. This ensures that useful trailing words are not abandoned simply
because they did not appear in enough partial snapshots.

### 5.4 Why This Matters

The architecture begins useful work before the full sentence is complete, but
does not allow unstable recognition to propagate through every downstream
stage. This is the central latency-quality balance of the solution.

---

## 6. AI Model Portfolio

### Vietnamese Speech Recognition

Vietnamese input uses GIPFormer, a compact transducer-based ASR model operating
through an INT8 ONNX runtime. It is selected for its Vietnamese recognition
capability, noise robustness, and edge-oriented execution.

### English Speech Recognition

English input uses SenseVoiceSmall through a locally exported and quantized
ONNX runtime. The deployment package is validated before startup so the system
does not silently accept a PyTorch-only or incomplete model directory.

### Machine Translation

Both language directions use VietAI EnViT5. The production-preferred runtime is
CTranslate2 with INT8 computation and greedy decoding.

Using one bilingual translation model simplifies deployment and terminology
management while keeping the runtime footprint lower than maintaining separate
models for each direction.

### Vietnamese Speech Synthesis

Vietnamese output uses the fine-tuned Kokoro Vietnamese model with the
**hung_thinh** male voice.

The fine-tuned model is retained because voice naturalness and Vietnamese
pronunciation quality are more important than changing runtimes without
evidence that quality is preserved.

### English Speech Synthesis

English output uses Kokoro with the **am_adam** male voice. The current laptop
runtime uses PyTorch CUDA on its RTX 3050.

An INT8 ONNX model was evaluated on CPU, but the quantized graph was slower than
PyTorch on the tested hardware. CUDA PyTorch was therefore selected for live
speech synthesis.

This is an important engineering lesson: quantization reduces model size, but
does not guarantee lower latency on every processor or runtime.

---

## 7. Latency-Aware TTS Design

TTS is optional and never blocks recognition or translation.

When Voice is enabled:

- Only committed translated text is eligible for synthesis.
- Requests are processed in conversation order.
- Duplicate text for the same segment is ignored.
- Small adjacent chunks may be combined before synthesis.
- The queue has a strict maximum size.
- Old pending work may be discarded if the conversation advances too far.
- Selecting Silence cancels queued requests and stops current playback.

The browser uses a single-flight request policy. It does not submit several TTS
requests simultaneously to a backend that can only run one inference safely.
This prevents lock waiting from being incorrectly perceived as model latency.

Only one output-language TTS model is kept active at a time. When the direction
changes, the target-language voice is prepared and the previous language model
is released. This lowers memory pressure on constrained devices.

---

## 8. Runtime Selection and Hardware Awareness

OneVoice does not assume that one inference backend is always best.

For English TTS, the device performs a local benchmark and compares the
measured Real-Time Factor against a configured acceptance threshold.

- If INT8 ONNX performs well enough, it remains active.
- If it is slower, the system selects warmed PyTorch.
- The decision is cached against the processor and model signature.
- Calibration is repeated only when the hardware, model, thread configuration,
  or performance threshold changes.

On the current Ryzen 7 6800H and RTX 3050 laptop:

- Kokoro INT8 ONNX measured approximately RTF 1.6.
- CPU PyTorch measured approximately RTF 0.28 to 0.30.
- CUDA PyTorch subsequently measured approximately RTF 0.07 to 0.08 for
  representative warmed speech chunks.
- CUDA PyTorch is therefore the active low-latency runtime for this device.

PyTorch is explicitly limited to the eight physical CPU cores. Before this
control was added, global thread-pool contention caused TTS to run two to four
times slower inside the complete application than in isolated benchmarks.

---

## 9. Measured Proof-of-Concept Performance

The following results are local measurements from the current laptop and should
be treated as engineering evidence, not universal hardware guarantees.

### Core Translation Pipeline

Representative end-to-end segment measurements:

| Direction | Example total latency |
|---|---:|
| Vietnamese to English | Approximately 267 ms |
| English to Vietnamese | Approximately 515 ms |

These measurements include audio conditioning, ASR, and MT for the tested
segments. They do not require TTS to finish before translated text is shown.

### Speech Recognition

The quantized SenseVoice ONNX runtime processed a 7.18-second English sample in
approximately 274 ms, corresponding to an RTF of about 0.038.

### TTS Before Optimization

Recent combined logs contained 30 speech chunks:

| Metric | Result |
|---|---:|
| Average synthesis-to-play time | 1,942 ms |
| 95th percentile | 3,284 ms |
| Maximum | 3,342 ms |

The main causes were:

- PyTorch thread-pool contention.
- Multiple browser requests waiting behind one backend inference lock.
- Duplicate submissions for the same segment.
- Both language TTS models competing for limited memory.

### TTS After Optimization

| Output language | Male voice | Average inference | Average RTF |
|---|---|---:|---:|
| English | am_adam | Approximately 188 ms | 0.067 |
| Vietnamese | hung_thinh | Approximately 348 ms | 0.082 |

Both directions remain below RTF 1.0 in the measured tests.

Model preparation after changing direction currently takes several seconds, but
it occurs when Voice is enabled or the direction changes, rather than delaying
every translated chunk.

---

## 10. Reliability and Observability

Production behavior cannot be improved using model logs alone. OneVoice records
both sides of the interaction:

### Browser Events

- Audio transmission.
- WebSocket state.
- Partial and committed rendering.
- TTS queue insertion.
- Deduplication and chunk coalescing.
- Synthesis response time.
- Playback start and completion.
- User cancellation.

### Backend Events

- Segment acceptance and processing.
- ASR and MT completion.
- Selected TTS engine.
- TTS inference duration.
- Audio duration.
- Real-Time Factor.
- Runtime preparation duration.
- Direction and segment identity.

These events share session and segment identifiers. A failed or delayed user
experience can therefore be traced across the browser, network boundary,
inference runtime, and playback queue.

---

## 11. Edge Deployment Assessment

### Current Strengths

- Fully local inference during operation.
- Quantized ASR and MT paths.
- RTF below 1.0 in measured recognition, translation, and selected TTS paths.
- Bounded queues and backpressure behavior.
- Runtime fallback when an optimized backend is unavailable or slower.
- Language-specific model lifecycle management.
- No requirement to send conversational audio to a cloud provider.

### Current Constraints

- The proof of concept still uses laptop-class memory and CPU resources.
- TTS model switching introduces a preparation delay.
- Partial ASR is based on repeated snapshots rather than a fully stateful
  token-streaming decoder.
- Vietnamese TTS does not yet have a quality-validated optimized ONNX export.
- Browser audio capture should eventually move from legacy processing callbacks
  to a modern low-latency audio worklet.

### Edge Suitability

The architecture is suitable for edge deployment, but the final hardware target
should provide:

- At least eight modern CPU cores or an equivalent accelerator.
- Sufficient memory for ASR, MT, and one active TTS model.
- Fast local storage for model startup.
- Reliable microphone and audio output drivers.
- Thermal capacity for continuous inference.

A smaller target device may require a reduced ASR/TTS portfolio, more aggressive
quantization, or dedicated NPU/GPU execution.

---

## 12. Business and Operational Value

OneVoice is applicable where privacy, connectivity, or response time makes
cloud translation unsuitable:

- Factories and industrial maintenance.
- Hospitals and emergency coordination.
- Field service and logistics.
- Tourism and hospitality.
- Government and public services.
- Security-sensitive workplaces.
- Remote communities and low-connectivity locations.

The product value is not limited to translation accuracy. Local execution also
provides:

- Predictable operating cost.
- Reduced exposure of private conversations.
- Continued service during network outages.
- Lower dependency on external APIs.
- Easier deployment in controlled environments.

---

## 13. Engineering Decisions and Lessons

### Optimize the Complete System

An isolated model benchmark is not enough. The same model can become much slower
when ASR, MT, TTS, and multiple native thread pools share one process.

### Measure Quantization

INT8 reduced the English TTS model size substantially, but did not improve
latency on the tested CPU. Runtime selection must be evidence-based.

### Keep Optional Work Outside the Critical Path

Translated text should appear without waiting for speech synthesis. TTS can then
be prepared, queued, played, or cancelled independently.

### Stability Is More Valuable Than Maximum Update Frequency

Users benefit from immediate partial text, but translation and speech should be
based on stable content. This produces a calmer and more trustworthy experience.

### Backlog Is a Latency Problem

Even a fast model feels slow when requests accumulate. Deduplication, bounded
queues, coalescing, and single-flight inference are as important as model speed.

---

## 14. Recommended Next Steps

### Near Term

- Evaluate male voices with native-speaker MOS testing.
- Build a fixed bilingual latency and quality benchmark suite.
- Measure first-audio latency separately from total TTS generation time.
- Add automatic reporting for average, median, p95, and worst-case latency.
- Test under industrial noise and varied microphone distances.

### Model and Runtime Research

- Export and validate the Vietnamese fine-tuned Kokoro model to ONNX.
- Compare FP32, FP16, INT8, and hardware-specific execution providers.
- Evaluate true stateful streaming ASR models.
- Test incremental translation policies using semantic boundaries rather than
  word count alone.

### Product Readiness

- Package model assets for fully offline installation.
- Add device health, temperature, memory, and model readiness monitoring.
- Create operator-friendly recovery behavior for microphone or audio failures.
- Conduct long-running stability tests.
- Define supported hardware tiers and expected performance per tier.

---

## 15. Conclusion

OneVoice Edge demonstrates that responsive bilingual voice translation can be
delivered locally without waiting for complete sentences and without translating
every unstable word.

Its key contribution is the coordination policy:

- Show partial speech immediately.
- Commit only stable text.
- Translate committed chunks.
- Render text before optional speech completes.
- Keep TTS ordered, bounded, measurable, and cancellable.
- Select runtimes according to real device performance.

This creates a practical foundation for an offline translation product that can
be improved through measurable model, hardware, and user-experience iterations.
