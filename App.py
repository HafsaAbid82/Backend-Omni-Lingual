import os, torch, librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

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
                if y.ndim > 1:
                    y = y.mean(axis=1) 
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
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

def make_prompt_examples():
    examples = []
    path = os.path.join(os.path.dirname(__file__), "Sample1.mp3")
    y, sr = load_audio_any_format(path)
    samples_per_chunk = 20 * sr  
    chunks = [y[i:i + samples_per_chunk] for i in range(0, len(y), samples_per_chunk)]
    for chunk in chunks:
        if len(chunk) / sr > 40:
            raise ValueError("Prompt chunk exceeds 40s limit")
        examples.append({
            "waveform": chunk,
            "sample_rate": 16000,
            "transcription": "دنیا کے نیک ترین نون مسلم جو نماز نہیں پڑھتا تھا یا مسلمان بھی جس نے ہیومن رائٹس پہ ساری زندگی کام کیا بڑی بڑی ارگنائزیشنز بنائیں بہت لوگوں کو کھانا کھلایا بہت لوگوں کی زندگیاں بچائیں ٹھیک ہے وہ نماز پڑھنے والا نہیں تھا لا حول ولا  یہ شخص اس دن نہیں بچ سکے گا بتا اگر یہی ٹیچنگ آج کے لیسن سے لے جائیں نا تو یہ ٹیچنگ بھی بہت ہے کہ نماز کچھ ہو جائے زندگی کا اصول بنا لے انسان اور پھر دیکھا جائے گا کہ کیا صلو کما رأیتمونی والصلی  تھا کہ نہیں تھا کیا اللہ کے نبی کے طریقے کے مطابق تھا کہ نہیں تھا نماز ہم بھی جب پڑھا کرتے تھے اور ہم اسی طریقے سے پڑھتے تھے جس طرح ماں باپ نے بتایا ہوا تھا ان کو جس طرح کرتے دیکھتے تھے ہم نے کبھی بودر نہیں کیا کہ وہ کتاب اٹھا کے دیکھیں اللہ کے نبی صلی اللہ علیہ وسلم نے کیسے نماز پڑھی اور یقین کریں جب اٹھا کے دیکھا کہ اللہ کہ نبی صلی اللہ علیہ وسلم کی نماز ایسی تھی تو شرم آئی کہ یار ہماری نماز یہ نماز ہے اصل نماز کی یہ ہے ہی نہیں یہ مس کر رہے ہیں ہم نماز میں کتاب پڑھیں آپ لوگ  جو دیکھ رہے ہیں نا صفت  صلاۃ النبی ہے یا آپ دار السلام چلے جائیں دار الہدی چلے جائیں نماز نبوی ہے اٹھائیں نماز نبوی لیں اور وہ پڑھیں کہ اللہ کے نبی کی نماز تھی کیسی نماز ٹھیک ہے کیونکہ اس بندے کے بارے میں سوچیں جس نے ساری زندگی نماز پڑھی اور وہ نماز اس کو آخرت میں جا کے پتہ چلے گا کہ جب جب اس نے وہ نماز پڑھی وہ اٹھا کے اس کے منہ پہ ماری اور اس کی ایک نمازی مثال کے طور پہ قبول نہیں ایسے لوگ ہوں گے ہوں گے ایسے لوگ لیکن نماز اس طرح نہیں پڑی جس طرح سے پڑھنے کا صحیح طریقہ تھا"  # replace with actual matching text
        })
    return examples

PROMPT_EXAMPLES = make_prompt_examples()
app = FastAPI(title="Omnilingual ASR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-omnilingual.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def transcribe_bulk(audio_paths, prompt_examples=None):
    if pipeline is None:
        return {"error": "Pipeline not initialized"}
    if not audio_paths:
        return {"error": "No audio files provided"}

    all_inputs, chunk_map = [], []
    if prompt_examples:
        for ex in prompt_examples:
            all_inputs.append({
                "waveform": ex["waveform"],
                "sample_rate": ex["sample_rate"]
            })

    for audio_path in audio_paths:
        y, sr = load_audio_any_format(audio_path, target_sr=16000)
        samples_per_chunk = CHUNK_LENGTH_S * sr
        chunks = [y[i:i + samples_per_chunk] for i in range(0, len(y), samples_per_chunk)]
        for idx, chunk in enumerate(chunks):
            if len(chunk) / sr > 40:
                raise ValueError("Audio chunk exceeds 40s limit")
            all_inputs.append({"waveform": chunk, "sample_rate": 16000})
            chunk_map.append((str(audio_path), idx))

    print(f"Processing {len(all_inputs)} chunks from {len(audio_paths)} files...")

    results = pipeline.transcribe(
        all_inputs,
        batch_size=BATCH_SIZE,
        lang=["urd_Arab"] * len(all_inputs)
    )

    offset = len(prompt_examples or [])
    output = {}
    for (audio_path, _), text in zip(chunk_map, results[offset:]):
        output.setdefault(audio_path, []).append(text)

    return {path: " ".join(texts) for path, texts in output.items()}

@app.post("/api/predict")
async def predict(files: List[UploadFile] = File(...), use_prompts: bool = True):
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

        prompt_examples = PROMPT_EXAMPLES if use_prompts else None
        results = transcribe_bulk(audio_paths, prompt_examples=prompt_examples)

        return {"results": results, "prompts_used": use_prompts}
    finally:
        for path in audio_paths:
            if os.path.exists(path):
                os.remove(path)

@app.get("/")
def root():
    return {"message": "Omnilingual-ASR Backend is running."}
