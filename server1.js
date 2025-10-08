import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import decode from "audio-decode";
import audiobufferToWav from "audiobuffer-to-wav";
import { createClient } from "@deepgram/sdk";

import { parseStringPromise } from "xml2js";
import sdk from "microsoft-cognitiveservices-speech-sdk";
import WavEncoder from "wav-encoder";

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = 3000;

app.use(express.json());
app.use(express.static("public"));

const SAMPLE_RATE = 22050;

/* ---------------- STEP 1: DEEPGRAM ---------------- */
function normalizeAudio(floatArray) {
  let maxAmp = 0;
  for (let i = 0; i < floatArray.length; i++) maxAmp = Math.max(maxAmp, Math.abs(floatArray[i]));
  if (maxAmp > 0) for (let i = 0; i < floatArray.length; i++) floatArray[i] /= maxAmp;
  return floatArray;
}

async function exportSpeakerFiles(utterances, audioBuffer, outDir) {
  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const totalSamples = audioBuffer.length;

  const mixedChannelData = new Float32Array(totalSamples);
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < totalSamples; i++) mixedChannelData[i] += channelData[i] / numChannels;
  }

  const speakers = {};
  utterances.forEach(u => {
    if (!speakers[u.speaker]) speakers[u.speaker] = [];
    speakers[u.speaker].push(u);
  });

  const files = [];
  for (const [speaker, segments] of Object.entries(speakers)) {
    const speakerBufferData = new Float32Array(totalSamples);
    segments.forEach(seg => {
      const startSample = Math.floor(seg.start * sampleRate);
      const endSample = Math.floor(seg.end * sampleRate);
      for (let i = startSample; i < endSample; i++) speakerBufferData[i] = mixedChannelData[i];
    });
    normalizeAudio(speakerBufferData);
    const speakerBuffer = { length: totalSamples, numberOfChannels: 1, sampleRate, getChannelData: () => speakerBufferData };
    const wavData = audiobufferToWav(speakerBuffer);
    const filename = `speaker${speaker}.wav`;
    fs.writeFileSync(path.join(outDir, filename), Buffer.from(wavData));
    files.push(filename);
  }
  return files;
}

function formatTimeSRT(seconds) {
  const date = new Date(seconds * 1000);
  const hh = String(date.getUTCHours()).padStart(2, "0");
  const mm = String(date.getUTCMinutes()).padStart(2, "0");
  const ss = String(date.getUTCSeconds()).padStart(2, "0");
  const ms = String(date.getUTCMilliseconds()).padStart(3, "0");
  return `${hh}:${mm}:${ss},${ms}`;
}

function generateXML(utterances) {
  let xml = `<?xml version="1.0" encoding="UTF-8"?>\n<captions>\n`;
  utterances.forEach(u => {
    xml += `  <caption speaker="speaker${u.speaker}" start="${u.start}" end="${u.end}">${escapeXML(u.transcript)}</caption>\n`;
  });
  xml += `</captions>`;
  return xml;
}

function escapeXML(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}

app.post("/upload", upload.single("audioFile"), async (req, res) => {
  const { apiKey } = req.body;
  const filePath = req.file.path;

  try {
    const deepgram = createClient(apiKey);
    const audio = fs.readFileSync(filePath);

    const { result } = await deepgram.listen.prerecorded.transcribeFile(audio, {
      model: "nova-2",
      diarize: true,
      punctuate: true,
      utterances: true,
    });

    const utterances = result.results.utterances;
    if (!utterances || utterances.length === 0) return res.json({ error: "No utterances detected." });

    const audioBuffer = await decode(audio);
    const outDir = path.join(process.cwd(), "public", "outputs");
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    const files = await exportSpeakerFiles(utterances, audioBuffer, outDir);
    const srtFile = "captions.srt";
    const xmlFile = "captions.xml";
    fs.writeFileSync(path.join(outDir, srtFile), "", "utf-8"); // blank SRT, will regenerate after TTS
    fs.writeFileSync(path.join(outDir, xmlFile), generateXML(utterances), "utf-8");
    files.push(srtFile, xmlFile);

    res.json({ files });
  } catch (err) {
    res.json({ error: err.message });
  } finally {
    fs.unlinkSync(filePath);
  }
});

/* ---------------- STEP 2: AZURE TTS ---------------- */
/* ---------------- STEP 2: AZURE TTS ---------------- */
const SPEECH_KEY = "EtDghUTrcJ9vtiuKHxKXa0dhhNnTUIVx4Xk8n5JYjBNDAGwyqc49JQQJ99BIACYeBjFXJ3w3AAAYACOGZsF9";
const SPEECH_REGION = "eastus";

// Synthesize text using Azure TTS
function synthesizeAzure(text, voiceName) {
  return new Promise((resolve, reject) => {
    const speechConfig = sdk.SpeechConfig.fromSubscription(SPEECH_KEY, SPEECH_REGION);
    speechConfig.speechSynthesisVoiceName = voiceName;
    speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm;

    const synthesizer = new sdk.SpeechSynthesizer(speechConfig);
    synthesizer.speakTextAsync(
      text,
      result => {
        synthesizer.close();
        if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) {
          const buffer = Buffer.from(result.audioData);
          const int16 = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
          const float32 = new Float32Array(int16.length);
          for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
          resolve(float32.length ? float32 : new Float32Array(1)); // fallback if empty
        } else reject(result.errorDetails);
      },
      error => { synthesizer.close(); reject(error); }
    );
  });
}

// Generate silence of given duration
function silence(durationSec) {
  const length = Math.max(1, Math.floor(durationSec * SAMPLE_RATE)); // at least 1 sample
  return new Float32Array(length);
}

// Encode single-channel Float32Array to WAV
async function encodeToWav(float32, filePath) {
  if (!float32 || float32.length === 0) float32 = new Float32Array(1);
  const audioData = { sampleRate: SAMPLE_RATE, channelData: [float32] };
  const buffer = await WavEncoder.encode(audioData);
  fs.writeFileSync(filePath, Buffer.from(buffer));
}

// Encode stereo WAV from two Float32Arrays
async function encodeCombinedWav(track0, track1, filePath) {
  const maxLength = Math.max(track0.length, track1.length);
  const left = new Float32Array(maxLength);
  const right = new Float32Array(maxLength);
  left.set(track0);
  right.set(track1);
  const buffer = await WavEncoder.encode({ sampleRate: SAMPLE_RATE, channelData: [left, right] });
  fs.writeFileSync(filePath, Buffer.from(buffer));
}

// Generate SRT purely based on TTS durations
function generateSRTFromTTS(captions, speechData, speakerNames) {
  let currentTime = 0;
  let srt = "";

  for (let i = 0; i < captions.length; i++) {
    const c = captions[i];
    const float32 = speechData[i];
    const duration = float32.length / SAMPLE_RATE;

    const start = formatTimeSRT(currentTime);
    const end = formatTimeSRT(currentTime + duration);
    const name = speakerNames[c.speaker] || c.speaker;

    srt += `${i + 1}\n${start} --> ${end}\n${name}: ${c.text.trim()}\n\n`;
    currentTime += duration;
  }

  return srt;
}

// POST /generate-tts endpoint
app.post("/generate-tts", async (req, res) => {
  const { xmlFile, speaker0Name, speaker1Name, speaker0Voice, speaker1Voice } = req.body;
  const xmlPath = path.join(process.cwd(), "public", "outputs", xmlFile);
  if (!fs.existsSync(xmlPath)) return res.json({ error: "XML not found" });

  try {
    const xml = fs.readFileSync(xmlPath, "utf-8");
    const result = await parseStringPromise(xml);
    const captions = result.captions.caption.map(c => ({
      speaker: c.$.speaker,
      start: parseFloat(c.$.start),
      end: parseFloat(c.$.end),
      text: c._
    }));

    const speakerVoices = { speaker0: speaker0Voice, speaker1: speaker1Voice };
    const speakerNames = { speaker0: speaker0Name, speaker1: speaker1Name };

    // Synthesize TTS for each caption
    const speechData = await Promise.all(
      captions.map(async c => {
        const data = await synthesizeAzure(c.text.trim(), speakerVoices[c.speaker]);
        return data || new Float32Array(1);
      })
    );

    // Build separate speaker tracks (no gaps from original Deepgram timing)
    let tracks = { speaker0: [], speaker1: [] };
    for (let i = 0; i < captions.length; i++) {
      const c = captions[i];
      const other = c.speaker === "speaker0" ? "speaker1" : "speaker0";
      const float32 = speechData[i];
      tracks[c.speaker].push(float32);
      tracks[other].push(silence(float32.length / SAMPLE_RATE));
    }

    // Merge tracks to single Float32Array per speaker
    const merged0 = Float32Array.from(tracks.speaker0.flatMap(arr => Array.from(arr || [])));
    const merged1 = Float32Array.from(tracks.speaker1.flatMap(arr => Array.from(arr || [])));
    const outDir = path.join(process.cwd(), "public", "outputs");

    await encodeToWav(merged0, path.join(outDir, "speaker0.wav"));
    await encodeToWav(merged1, path.join(outDir, "speaker1.wav"));
    await encodeCombinedWav(merged0, merged1, path.join(outDir, "combined.wav"));

    // Generate SRT using TTS timeline
    const newSRT = generateSRTFromTTS(captions, speechData, speakerNames);
    const srtFile = "combined.srt";
    fs.writeFileSync(path.join(outDir, srtFile), newSRT, "utf-8");

    res.json({ files: ["speaker0.wav", "speaker1.wav", "combined.wav", srtFile] });

  } catch (err) {
    res.json({ error: err.message });
  }
});

app.get("/download/:filename", (req, res) => {
  const filePath = path.join(process.cwd(), "public", "outputs", req.params.filename);
  res.download(filePath);
});

app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
