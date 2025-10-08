// server.js (Option A — fixed timeline anchoring, per-caption splicing, pad/truncate TTS)
import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import decode from "audio-decode";          // decode audio file to AudioBuffer-like
import { createClient } from "@deepgram/sdk";
import { parseStringPromise } from "xml2js";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import WavEncoder from "wav-encoder";
import audiobufferToWav from "audiobuffer-to-wav";

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = 3000;

// final output sample rate (choose 22050 or 24000 etc). We'll use 22050 for smaller files.
const SAMPLE_RATE = 22050;

// Azure keys (replace)
const SPEECH_KEY = "EtDghUTrcJ9vtiuKHxKXa0dhhNnTUIVx4Xk8n5JYjBNDAGwyqc49JQQJ99BIACYeBjFXJ3w3AAAYACOGZsF9";
const SPEECH_REGION = "eastus";

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));

/* ---------- Helpers ---------- */

function clamp(v, a = -1, b = 1) { return Math.max(a, Math.min(b, v)); }

// normalize Float32Array in-place to -0.98..0.98
function normalizeInPlace(arr) {
  let maxAmp = 0;
  for (let i = 0; i < arr.length; i++) maxAmp = Math.max(maxAmp, Math.abs(arr[i]));
  if (maxAmp < 1e-8) return arr;
  const scale = 0.98 / maxAmp;
  for (let i = 0; i < arr.length; i++) arr[i] = clamp(arr[i] * scale);
  return arr;
}

// linear resample Float32Array from srcRate -> dstRate
function resampleLinear(src, srcRate, dstRate) {
  if (!src || src.length === 0) return new Float32Array(1);
  if (srcRate === dstRate) return src;
  const srcLen = src.length;
  const dstLen = Math.max(1, Math.round(srcLen * dstRate / srcRate));
  const out = new Float32Array(dstLen);
  const factor = (srcLen - 1) / (dstLen - 1);
  for (let i = 0; i < dstLen; i++) {
    const pos = i * factor;
    const i0 = Math.floor(pos);
    const i1 = Math.min(srcLen - 1, i0 + 1);
    const t = pos - i0;
    out[i] = src[i0] * (1 - t) + src[i1] * t;
  }
  return out;
}

function formatTimeSRT(seconds) {
  const date = new Date(Math.round(seconds * 1000));
  const hh = String(date.getUTCHours()).padStart(2, "0");
  const mm = String(date.getUTCMinutes()).padStart(2, "0");
  const ss = String(date.getUTCSeconds()).padStart(2, "0");
  const ms = String(date.getUTCMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss},${ms}`;
}

function escapeXML(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}

function generateXMLFromUtterances(utterances) {
  let xml = `<?xml version="1.0" encoding="UTF-8"?>\n<captions>\n`;
  utterances.forEach(u => {
    xml += `  <caption speaker="speaker${u.speaker}" start="${u.start}" end="${u.end}">${escapeXML(u.transcript)}</caption>\n`;
  });
  xml += `</captions>\n`;
  return xml;
}

/* ---------- Deepgram upload & diarization (Step 1) ---------- */

async function exportSpeakerFilesFromAudioBuffer(utterances, audioBuffer, outDir) {
  // Create per-speaker full-length waveform files (mono) using the decoded audioBuffer mixdown
  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const totalSamples = audioBuffer.length;

  // mixdown to mono
  const mixed = new Float32Array(totalSamples);
  for (let ch = 0; ch < numChannels; ch++) {
    const chd = audioBuffer.getChannelData(ch);
    for (let i = 0; i < totalSamples; i++) mixed[i] += chd[i] / numChannels;
  }

  const speakers = {};
  utterances.forEach(u => {
    const key = `speaker${u.speaker}`;
    if (!speakers[key]) speakers[key] = [];
    speakers[key].push(u);
  });

  const files = [];
  for (const [speaker, segs] of Object.entries(speakers)) {
    const buf = new Float32Array(totalSamples); // zeros (silence)
    segs.forEach(seg => {
      const s = Math.max(0, Math.floor(seg.start * sampleRate));
      const e = Math.min(totalSamples, Math.floor(seg.end * sampleRate));
      for (let i = s; i < e; i++) buf[i] = mixed[i];
    });
    normalizeInPlace(buf);
    // encode at original sampleRate so slicing is precise
    const audioData = { sampleRate, channelData: [buf] };
    const wavBuf = audiobufferToWav({ length: totalSamples, numberOfChannels: 1, sampleRate, getChannelData: () => buf });
    const filename = `${speaker}.wav`;
    fs.writeFileSync(path.join(outDir, filename), Buffer.from(wavBuf));
    files.push(filename);
  }
  return files;
}

app.post("/upload", upload.single("audioFile"), async (req, res) => {
  const apiKey = req.body.apiKey || req.body.dgKey || req.body.dgkey;
  if (!apiKey) {
    if (req.file && req.file.path) try { fs.unlinkSync(req.file.path); } catch (e) {}
    return res.json({ error: "Missing Deepgram API key (apiKey)" });
  }
  if (!req.file) return res.json({ error: "No audio file uploaded (field name: audioFile)" });

  const uploadedPath = req.file.path;
  try {
    const dg = createClient(apiKey);
    const fileBuf = fs.readFileSync(uploadedPath);

    const { result } = await dg.listen.prerecorded.transcribeFile(fileBuf, {
      model: "nova-2",
      diarize: true,
      punctuate: true,
      utterances: true
    });

    const utterances = result?.results?.utterances || [];
    if (!utterances.length) {
      fs.unlinkSync(uploadedPath);
      return res.json({ error: "No utterances detected from Deepgram" });
    }

    // decode original audio to AudioBuffer-like
    const decoded = await decode(fileBuf); // { sampleRate, length, numberOfChannels, getChannelData }
    const outDir = path.join(process.cwd(), "public", "outputs");
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    // export per-speaker full-length files (helpful for debugging)
    const exportedFiles = await exportSpeakerFilesFromAudioBuffer(utterances, decoded, outDir);

    // store original uploaded mixed audio (needed for exact caption slicing)
    fs.writeFileSync(path.join(outDir, "original_uploaded.wav"), fileBuf);

    // write captions.xml (we'll use it as canonical)
    const xml = generateXMLFromUtterances(utterances);
    fs.writeFileSync(path.join(outDir, "captions.xml"), xml, "utf-8");

    // blank placeholder srt
    fs.writeFileSync(path.join(outDir, "captions.srt"), "", "utf-8");

    res.json({ files: exportedFiles.concat(["captions.srt", "captions.xml", "original_uploaded.wav"]), xmlFile: "captions.xml" });
  } catch (err) {
    console.error("upload error", err);
    res.json({ error: err.message || String(err) });
  } finally {
    try { fs.unlinkSync(uploadedPath); } catch (e) {}
  }
});

/* ---------- Azure TTS + per-caption timeline assembly (Step 2) ---------- */

function synthesizeAzure(text, voiceName) {
  return new Promise((resolve, reject) => {
    // if empty text, return tiny silence
    if (!text || text.trim().length === 0) return resolve(new Float32Array(1));

    const speechConfig = sdk.SpeechConfig.fromSubscription(SPEECH_KEY, SPEECH_REGION);
    if (voiceName) speechConfig.speechSynthesisVoiceName = voiceName;
    // request raw 24k PCM mono
    speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm;

    const synthesizer = new sdk.SpeechSynthesizer(speechConfig);
    synthesizer.speakTextAsync(
      text,
      result => {
        synthesizer.close();
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          const buf = Buffer.from(result.audioData);
          // convert int16 PCM -> Float32
          const int16 = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 2));
          const f32 = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) f32[i] = clamp(int16[i] / 32768);
          resolve(f32);
        } else {
          reject(result.errorDetails || "Azure TTS failed");
        }
      },
      err => { synthesizer.close(); reject(err); }
    );
  });
}

// extract absolute segment from original uploaded mixed audio (not per-speaker file)
// returns Float32Array sampled at original sampleRate
async function extractSegmentFromOriginalMixed(startSec, endSec) {
  const outDir = path.join(process.cwd(), "public", "outputs");
  const origPath = path.join(outDir, "original_uploaded.wav");
  if (!fs.existsSync(origPath)) return new Float32Array(1);

  const buf = fs.readFileSync(origPath);
  const audioBuffer = await decode(buf); // has sampleRate and getChannelData
  const sampleRate = audioBuffer.sampleRate;
  const startSample = Math.max(0, Math.floor(startSec * sampleRate));
  const endSample = Math.min(audioBuffer.length, Math.floor(endSec * sampleRate));
  const numCh = audioBuffer.numberOfChannels;
  // mixdown to mono for extraction
  const out = new Float32Array(Math.max(1, endSample - startSample));
  for (let ch = 0; ch < numCh; ch++) {
    const chd = audioBuffer.getChannelData(ch).subarray(startSample, endSample);
    for (let i = 0; i < out.length; i++) out[i] += (chd[i] || 0) / numCh;
  }
  return out; // sampled at sampleRate (caller must resample)
}

async function encodeFloat32ToWav(float32, sampleRate, channelCount, filePath) {
  // channelCount 1 for mono, 2 for stereo
  // build channelData
  const channelData = [];
  if (channelCount === 1) channelData.push(float32);
  else {
    // assume interleaved? we expect left/right separate arrays; caller should handle stereo composition separately
    throw new Error("This helper expects mono Float32 and channelCount=1");
  }
  const buffer = await WavEncoder.encode({ sampleRate, channelData });
  fs.writeFileSync(filePath, Buffer.from(buffer));
}

// pad or truncate a Float32Array to targetLength (samples)
function padOrTruncate(arr, targetLen) {
  if (arr.length === targetLen) return arr;
  if (arr.length > targetLen) return arr.subarray(0, targetLen);
  const out = new Float32Array(targetLen);
  out.set(arr, 0);
  // remaining are zeros (silence)
  return out;
}

app.post("/generate-tts", async (req, res) => {
  try {
    const { xmlFile, speaker0Name, speaker1Name, speaker0Voice, speaker1Voice } = req.body;
    if (!xmlFile) return res.json({ error: "xmlFile is required in body" });

    const outDir = path.join(process.cwd(), "public", "outputs");
    const xmlPath = path.join(outDir, xmlFile);
    if (!fs.existsSync(xmlPath)) return res.json({ error: "captions XML not found" });

    const xml = fs.readFileSync(xmlPath, "utf-8");
    const parsed = await parseStringPromise(xml);
    const captionsRaw = (parsed?.captions?.caption || []).map(c => ({
      speaker: c.$.speaker,
      start: parseFloat(c.$.start),
      end: parseFloat(c.$.end),
      text: (c._ || "").trim()
    }));

    if (!captionsRaw.length) return res.json({ error: "No captions found in XML" });

    // Determine full timeline length (in seconds) using last caption end
    const timelineSeconds = Math.max(...captionsRaw.map(c => c.end));
    const timelineSamples = Math.max(1, Math.ceil(timelineSeconds * SAMPLE_RATE));

    // Prepare two timeline buffers (mono)
    const timeline0 = new Float32Array(timelineSamples); // speaker0 placed at left
    const timeline1 = new Float32Array(timelineSamples); // speaker1 placed at right

    // For convenience, map voices/names
    const speakerVoices = { speaker0: speaker0Voice || "", speaker1: speaker1Voice || "" };
    const speakerNames = { speaker0: speaker0Name || "Speaker 0", speaker1: speaker1Name || "Speaker 1" };

    // We'll decode original uploaded audio once to read its sampleRate (but we will use extractSegment per-caption)
    const origPath = path.join(outDir, "original_uploaded.wav");
    if (!fs.existsSync(origPath)) return res.json({ error: "original_uploaded.wav missing — run /upload first" });

    // Process captions sequentially building timeline
    for (let i = 0; i < captionsRaw.length; i++) {
      const c = captionsRaw[i];
      const speaker = c.speaker; // speaker0 or speaker1
      const startSample = Math.max(0, Math.floor(c.start * SAMPLE_RATE));
      const captionDurSec = Math.max(0.001, c.end - c.start);
      const captionLen = Math.max(1, Math.round(captionDurSec * SAMPLE_RATE));
      let segmentResampled;

      if (!speakerVoices[speaker]) {
        // KEEP ORIGINAL: extract from original uploaded mixed audio for exact time range
        const origSegment = await extractSegmentFromOriginalMixed(c.start, c.end); // at original sampleRate
        // Need to know original sample rate -> decode briefly
        const origBuf = fs.readFileSync(origPath);
        const origBufferDecoded = await decode(origBuf);
        const origRate = origBufferDecoded.sampleRate;
        // resample original to SAMPLE_RATE
        segmentResampled = resampleLinear(origSegment, origRate, SAMPLE_RATE);
        // pad/truncate to captionLen to ensure strict alignment to caption duration
        segmentResampled = padOrTruncate(segmentResampled, captionLen);
      } else {
        // GENERATE TTS for this caption
        const ttsFloat = await synthesizeAzure(c.text || ".", speakerVoices[speaker]); // Azure returns 24k
        const AZURE_RATE = 24000; // matches Raw24Khz16BitMonoPcm requested
        const ttsResampled = resampleLinear(ttsFloat, AZURE_RATE, SAMPLE_RATE);
        // We must force ttsResampled to match the original caption duration (pad or truncate)
        segmentResampled = padOrTruncate(ttsResampled, captionLen);
      }

      // Write segmentResampled into the correct speaker timeline at absolute startSample
      if (startSample + segmentResampled.length > timelineSamples) {
        // clamp segment length if it would overflow due to rounding
        const allowed = Math.max(0, timelineSamples - startSample);
        if (allowed <= 0) continue;
        segmentResampled = segmentResampled.subarray(0, allowed);
      }

      const targetTimeline = speaker === "speaker0" ? timeline0 : timeline1;
      for (let s = 0; s < segmentResampled.length; s++) {
        // simple overwrite: segments shouldn't overlap for same speaker (Deepgram utterances are sequential)
        targetTimeline[startSample + s] += segmentResampled[s];
      }
      // For the other speaker we intentionally leave zeros (silence) — no need to fill with explicit silence
    }

    // Light normalization (prevent clipping)
    normalizeInPlace(timeline0);
    normalizeInPlace(timeline1);

    // Save outputs
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
    await encodeFloat32MonoToWavFile(timeline0, SAMPLE_RATE, path.join(outDir, "speaker0.wav"));
    await encodeFloat32MonoToWavFile(timeline1, SAMPLE_RATE, path.join(outDir, "speaker1.wav"));
    await encodeFloat32StereoToWavFile(timeline0, timeline1, SAMPLE_RATE, path.join(outDir, "combined.wav"));

    // generate SRT using original caption start/end times and names
    let srt = "";
    for (let i = 0; i < captionsRaw.length; i++) {
      const c = captionsRaw[i];
      const start = formatTimeSRT(c.start);
      const end = formatTimeSRT(c.end);
      const name = speakerNames[c.speaker] || c.speaker;
      srt += `${i + 1}\n${start} --> ${end}\n${name}: ${c.text}\n\n`;
    }
    const srtFile = "combined.srt";
    fs.writeFileSync(path.join(outDir, srtFile), srt, "utf-8");

    res.json({
      files: ["speaker0.wav", "speaker1.wav", "combined.wav", srtFile],
      urls: {
        speaker0: `/outputs/speaker0.wav`,
        speaker1: `/outputs/speaker1.wav`,
        combined: `/outputs/combined.wav`,
        srt: `/outputs/${srtFile}`
      }
    });

  } catch (err) {
    console.error("generate-tts error:", err);
    res.json({ error: err.message || String(err) });
  }
});

/* ---------- Small encode helpers (mono + stereo) ---------- */

async function encodeFloat32MonoToWavFile(float32, sampleRate, filePath) {
  // ensure at least 1 sample
  if (!float32 || float32.length === 0) float32 = new Float32Array(1);
  const buffer = await WavEncoder.encode({ sampleRate, channelData: [float32] });
  fs.writeFileSync(filePath, Buffer.from(buffer));
}

async function encodeFloat32StereoToWavFile(left, right, sampleRate, filePath) {
  const maxLen = Math.max(left.length, right.length);
  const L = new Float32Array(maxLen);
  const R = new Float32Array(maxLen);
  L.set(left);
  R.set(right);
  const buffer = await WavEncoder.encode({ sampleRate, channelData: [L, R] });
  fs.writeFileSync(filePath, Buffer.from(buffer));
}

/* ---------- download route for convenience ---------- */
app.get("/download/:filename", (req, res) => {
  const filePath = path.join(process.cwd(), "public", "outputs", req.params.filename);
  if (!fs.existsSync(filePath)) return res.status(404).send("Not found");
  res.download(filePath);
});

app.listen(PORT, () => console.log(`🚀 Server listening on http://localhost:${PORT}`));
