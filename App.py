import os, torch, librosa, asyncio, atexit
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

ACCURACY_MODEL_CARD = "omniASR_LLM_1B_v2"
CHUNK_LENGTH_S = 30
BATCH_SIZE = 16 
VALID_LANGS = ["urd-arab", "pnb-arab", "snd-arab", "pus-arab", "bal-arab"]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
def load_audio_any_format(path, target_sr=16000):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".wav", ".flac", ".ogg"]:
            y, sr = sf.read(path)
            if sr != target_sr:
                y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T
                sr = target_sr
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y, sr
        else:
            y, sr = librosa.load(path, sr=target_sr, mono=True)
            return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")
try:
    print("Initializing ASR Pipeline...")
    pipeline = ASRInferencePipeline(
        model_card=ACCURACY_MODEL_CARD,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Pipeline Ready.")
except Exception as e:
    print(f"Initialization Error: {e}")
    pipeline = None
app = FastAPI(title="Omnilingual ASR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-omnilingual.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/api/predict")
async def predict(files: List[UploadFile] = File(...)):
    if pipeline is None:
        return {"error": "Pipeline not initialized"}
    if not files:
        return {"error": "No audio files provided"}
    audio_paths = []
    try:
        for file in files:
            safe_name = os.path.basename(file.filename)
            tmp_path = os.path.join("/tmp", safe_name)
            with open(tmp_path, "wb") as f:
                f.write(await file.read())
            audio_paths.append(tmp_path)
        results = transcribe_bulk(audio_paths)
        return {"results": results }
    finally:
        for path in audio_paths:
            if os.path.exists(path):
                os.remove(path)

def transcribe_bulk(audio_paths):
    if pipeline is None:
        return {"error": "Pipeline not initialized"}
    if not audio_paths:
        return {"error": "No audio files provided"}
    all_inputs, chunk_map = [], []
    for audio_path in audio_paths:
        y, sr = load_audio_any_format(audio_path, target_sr=16000)
        samples_per_chunk = CHUNK_LENGTH_S * sr
        chunks = [y[i:i + samples_per_chunk] for i in range(0, len(y), samples_per_chunk)]
        for idx, chunk in enumerate(chunks):
            all_inputs.append({"waveform": chunk, "sample_rate": 16000})
            chunk_map.append((str(audio_path), idx))
    print(f"Processing {len(all_inputs)} chunks from {len(audio_paths)} files...")
    results = pipeline.transcribe(all_inputs, batch_size=BATCH_SIZE, lang=["urd-arab"] * len(all_inputs))
    output = {}
    for (audio_path, _), text in zip(chunk_map, results):
        output.setdefault(audio_path, []).append(text)
    return {path: " ".join(texts) for path, texts in output.items()}
@app.get("/")
def root():
    return {"message": "Omnilingual-ASR Backend is running."}  
