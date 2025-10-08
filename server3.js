// server.js — Improved smooth TTS + mixed original audio
import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import decode from "audio-decode";
import { createClient } from "@deepgram/sdk";
import { parseStringPromise } from "xml2js";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import WavEncoder from "wav-encoder";
import audiobufferToWav from "audiobuffer-to-wav";

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = 3000;

const SAMPLE_RATE = 22050;
const SPEECH_KEY = "EtDghUTrcJ9vtiuKHxKXa0dhhNnTUIVx4Xk8n5JYjBNDAGwyqc49JQQJ99BIACYeBjFXJ3w3AAAYACOGZsF9";
const SPEECH_REGION = "eastus";

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));

/* ---------- Helpers ---------- */

function clamp(v, a = -1, b = 1) { return Math.max(a, Math.min(b, v)); }
function normalizeInPlace(arr) {
  let maxAmp = 0;
  for (let i = 0; i < arr.length; i++) maxAmp = Math.max(maxAmp, Math.abs(arr[i]));
  if (maxAmp < 1e-8) return arr;
  const scale = 0.98 / maxAmp;
  for (let i = 0; i < arr.length; i++) arr[i] = clamp(arr[i] * scale);
  return arr;
}
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

/* ---------- Fade + crossfade helpers ---------- */

function applyFade(segment, fadeSamples = 128) {
  const len = segment.length;
  for (let i = 0; i < fadeSamples && i < len; i++) {
    segment[i] *= i / fadeSamples;           // fade-in
    segment[len - i - 1] *= i / fadeSamples; // fade-out
  }
  return segment;
}

function padOrTruncate(arr, targetLen) {
  if (arr.length === targetLen) return arr;
  if (arr.length > targetLen) return arr.subarray(0, targetLen);
  const out = new Float32Array(targetLen);
  out.set(arr, 0);
  return out;
}

/* ---------- Deepgram upload & diarization ---------- */

async function exportSpeakerFilesFromAudioBuffer(utterances, audioBuffer, outDir) {
  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const totalSamples = audioBuffer.length;
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
    const buf = new Float32Array(totalSamples);
    segs.forEach(seg => {
      const s = Math.max(0, Math.floor(seg.start * sampleRate));
      const e = Math.min(totalSamples, Math.floor(seg.end * sampleRate));
      for (let i = s; i < e; i++) buf[i] = mixed[i];
    });
    normalizeInPlace(buf);
    // const wavBuf = audiobufferToWav(buf, { sampleRate });
    const filename = `${speaker}.wav`;
    // fs.writeFileSync(path.join(outDir, filename), Buffer.from(wavBuf));
    await WavEncoder.encode({ sampleRate, channelData: [buf] })
  .then(buffer => fs.writeFileSync(path.join(outDir, filename), Buffer.from(buffer)));

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

    const decoded = await decode(fileBuf);
    const outDir = path.join(process.cwd(), "public", "outputs");
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    const exportedFiles = await exportSpeakerFilesFromAudioBuffer(utterances, decoded, outDir);

    fs.writeFileSync(path.join(outDir, "original_uploaded.wav"), fileBuf);
    const xml = generateXMLFromUtterances(utterances);
    fs.writeFileSync(path.join(outDir, "captions.xml"), xml, "utf-8");
    fs.writeFileSync(path.join(outDir, "captions.srt"), "", "utf-8");

    res.json({ files: exportedFiles.concat(["captions.srt", "captions.xml", "original_uploaded.wav"]), xmlFile: "captions.xml" });
  } catch (err) {
    console.error("upload error", err);
    res.json({ error: err.message || String(err) });
  } finally {
    try { fs.unlinkSync(uploadedPath); } catch (e) {}
  }
});

/* ---------- Azure TTS + timeline assembly ---------- */

function synthesizeAzure(text, voiceName) {
  return new Promise((resolve, reject) => {
    if (!text || text.trim().length === 0) return resolve(new Float32Array(1));
    const speechConfig = sdk.SpeechConfig.fromSubscription(SPEECH_KEY, SPEECH_REGION);
    if (voiceName) speechConfig.speechSynthesisVoiceName = voiceName;
    speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm;

    const synthesizer = new sdk.SpeechSynthesizer(speechConfig);
    synthesizer.speakTextAsync(
      text,
      result => {
        synthesizer.close();
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          const buf = Buffer.from(result.audioData);
          const int16 = new Int16Array(buf.buffer, buf.byteOffset, Math.floor(buf.length / 2));
          const f32 = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) f32[i] = clamp(int16[i] / 32768);
          resolve(f32);
        } else reject(result.errorDetails || "Azure TTS failed");
      },
      err => { synthesizer.close(); reject(err); }
    );
  });
}

async function extractSegmentFromOriginalMixed(startSec, endSec) {
  const outDir = path.join(process.cwd(), "public", "outputs");
  const origPath = path.join(outDir, "original_uploaded.wav");
  if (!fs.existsSync(origPath)) return new Float32Array(1);

  const buf = fs.readFileSync(origPath);
  const audioBuffer = await decode(buf);
  const sampleRate = audioBuffer.sampleRate;
  const startSample = Math.max(0, Math.floor(startSec * sampleRate));
  const endSample = Math.min(audioBuffer.length, Math.floor(endSec * sampleRate));
  const numCh = audioBuffer.numberOfChannels;
  const out = new Float32Array(Math.max(1, endSample - startSample));
  for (let ch = 0; ch < numCh; ch++) {
    const chd = audioBuffer.getChannelData(ch).subarray(startSample, endSample);
    for (let i = 0; i < out.length; i++) out[i] += (chd[i] || 0) / numCh;
  }
  return out;
}

async function encodeFloat32MonoToWavFile(float32, sampleRate, filePath) {
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

    const timelineSeconds = Math.max(...captionsRaw.map(c => c.end));
    const timelineSamples = Math.max(1, Math.ceil(timelineSeconds * SAMPLE_RATE));

    const timeline0 = new Float32Array(timelineSamples);
    const timeline1 = new Float32Array(timelineSamples);

    const speakerVoices = { speaker0: speaker0Voice || "", speaker1: speaker1Voice || "" };
    const speakerNames = { speaker0: speaker0Name || "Speaker 0", speaker1: speaker1Name || "Speaker 1" };

    const origPath = path.join(outDir, "original_uploaded.wav");
    if (!fs.existsSync(origPath)) return res.json({ error: "original_uploaded.wav missing — run /upload first" });

    for (let i = 0; i < captionsRaw.length; i++) {
      const c = captionsRaw[i];
      const speaker = c.speaker;
      const startSample = Math.max(0, Math.floor(c.start * SAMPLE_RATE));
      const captionDurSec = Math.max(0.001, c.end - c.start);
      const captionLen = Math.max(1, Math.round(captionDurSec * SAMPLE_RATE));
      let segmentResampled;

      if (!speakerVoices[speaker]) {
        const origSegment = await extractSegmentFromOriginalMixed(c.start, c.end);
        const origBuf = fs.readFileSync(origPath);
        const origBufferDecoded = await decode(origBuf);
        const origRate = origBufferDecoded.sampleRate;
        segmentResampled = resampleLinear(origSegment, origRate, SAMPLE_RATE);
        segmentResampled = padOrTruncate(segmentResampled, captionLen);
      } else {
        const ttsFloat = await synthesizeAzure(c.text || ".", speakerVoices[speaker]);
        const AZURE_RATE = 24000;
        const ttsResampled = resampleLinear(ttsFloat, AZURE_RATE, SAMPLE_RATE);
        segmentResampled = padOrTruncate(ttsResampled, captionLen);
      }

      // apply fade-in/out for smoothness
      segmentResampled = applyFade(segmentResampled, Math.floor(SAMPLE_RATE * 0.01)); // 10ms fade

      const targetTimeline = speaker === "speaker0" ? timeline0 : timeline1;
      const crossfadeSamples = Math.floor(SAMPLE_RATE * 0.02); // 20ms crossfade
      for (let s = 0; s < segmentResampled.length; s++) {
        if (s < crossfadeSamples && startSample + s < targetTimeline.length) {
          targetTimeline[startSample + s] =
            targetTimeline[startSample + s] * (1 - s / crossfadeSamples) +
            segmentResampled[s] * (s / crossfadeSamples);
        } else if (startSample + s < targetTimeline.length) {
          targetTimeline[startSample + s] += segmentResampled[s];
        }
      }
    }

    normalizeInPlace(timeline0);
    normalizeInPlace(timeline1);

    await encodeFloat32MonoToWavFile(timeline0, SAMPLE_RATE, path.join(outDir, "speaker0.wav"));
    await encodeFloat32MonoToWavFile(timeline1, SAMPLE_RATE, path.join(outDir, "speaker1.wav"));
    await encodeFloat32StereoToWavFile(timeline0, timeline1, SAMPLE_RATE, path.join(outDir, "combined.wav"));

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

app.get("/download/:filename", (req, res) => {
  const filePath = path.join(process.cwd(), "public", "outputs", req.params.filename);
  if (!fs.existsSync(filePath)) return res.status(404).send("Not found");
  res.download(filePath);
});

app.listen(PORT, () => console.log(`🚀 Server listening on http://localhost:${PORT}`));
