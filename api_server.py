from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response, FileResponse
from pydantic import BaseModel
from voxcpm import VoxCPM
import torch
import io

# 开启 TF32 加速 (针对 NVIDIA Ampere 及以上显卡)
torch.set_float32_matmul_precision('high')
import os
import uvicorn
import soundfile as sf
import base64
import json
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

import asyncio

app = FastAPI(title="VoxCPM2 All-In-One API")

# 启用 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型变量 & 并发锁
model = None
model_lock = asyncio.Lock()

def create_wav_header(sample_rate, num_channels, bits_per_sample, data_size=0x7FFFFFFF):
    import struct
    # RIFF header
    o = b'RIFF'
    o += struct.pack('<I', data_size + 36)
    o += b'WAVE'
    # fmt chunk
    o += b'fmt '
    o += struct.pack('<I', 16)
    o += struct.pack('<H', 1)  # PCM
    o += struct.pack('<H', num_channels)
    o += struct.pack('<I', sample_rate)
    o += struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8)
    o += struct.pack('<H', num_channels * bits_per_sample // 8)
    o += struct.pack('<H', bits_per_sample)
    # data chunk
    o += b'data'
    o += struct.pack('<I', data_size)
    return o

class OpenAISpeechRequest(BaseModel):
    model: str = "voxcpm-2"
    input: str
    voice: str = "default"         # 用于音色克隆 (Identity)
    voice_data: str = ""           # 用于上传音色克隆的音频 (Base64)
    prompt_text: str = ""          # 用于情感/语境参考的文本
    prompt_voice: str = "default"  # 用于情感/语境参考的音频 (Style)
    prompt_voice_data: str = ""    # 用于上传情感/语境参考的音频 (Base64)
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = True
    stream_format: str = "audio" 
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    seed: int = -1

@app.get("/")
async def get_index():
    return FileResponse("web_demo.html")

@app.on_event("startup")
def load_model():
    global model
    model_path = "./pretrained_models/VoxCPM2"
    if not os.path.exists("temp"):
        os.makedirs("temp")
    print(f"Loading model into memory from {model_path}...")
    model = VoxCPM.from_pretrained(model_path, load_denoiser=False)
    print("Model is ready.")

def get_ref_path(voice_name: str, voice_data: str = None, prefix: str = "ref"):
    if voice_data:
        # 如果有 Base64 数据，保存到临时文件
        import uuid
        file_path = os.path.join("temp", f"{prefix}_{uuid.uuid4()}.wav")
        try:
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(voice_data))
            return file_path
        except Exception as e:
            print(f"Error saving base64 data: {e}")
            return None

    if not voice_name or voice_name == "default":
        return None
        
    # 优先在 refvoice/ 目录下找，支持带不带 .wav 后缀
    potential_paths = [
        os.path.join("refvoice", f"{voice_name}.wav"),
        os.path.join("refvoice", voice_name),
        f"{voice_name}.wav",
        voice_name
    ]
    for p in potential_paths:
        if os.path.exists(p):
            return p
    return None

@app.post("/v1/audio/speech")
async def create_speech(request: OpenAISpeechRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    temp_files = []
    try:
        # 解析参考路径
        ref_path = get_ref_path(request.voice, request.voice_data, "voice")
        prompt_wav_path = get_ref_path(request.prompt_voice, request.prompt_voice_data, "prompt")
        prompt_text = request.prompt_text if request.prompt_text else None

        if request.voice_data: temp_files.append(ref_path)
        if request.prompt_voice_data: temp_files.append(prompt_wav_path)

        # 确定合成模式
        has_prompt = (prompt_wav_path and prompt_text)
        has_ref = (ref_path is not None)

        # 检查用户是否意图使用某种模式但缺少必要条件
        if request.prompt_voice != "default" or request.prompt_voice_data:
            if not has_prompt:
                raise HTTPException(status_code=400, detail="情感延续模式需要同时提供 [参考音频] 和 [参考文本]")
        
        if request.voice != "default" or request.voice_data:
            if not has_ref:
                raise HTTPException(status_code=400, detail=f"找不到指定的音色参考文件: {request.voice}")

        if has_ref and has_prompt:
            mode = "Ultimate Cloning"
        elif has_prompt:
            mode = "Continuation"
        elif has_ref:
            mode = "Voice Cloning"
        elif "(" in request.input and ")" in request.input:
            mode = "Voice Design"
        else:
            mode = "Zero-shot"
        
        print(f"--- New Request ---")
        print(f"Full Payload: {request.dict()}")
        print(f"Mode: [{mode}] | CFG: {request.cfg_value} | Steps: {request.inference_timesteps} | Seed: {request.seed}")
        print(f"Input Text: {request.input[:50]}{'...' if len(request.input) > 50 else ''}")

        # --- 流式逻辑 ---
        if request.stream:
            async def generate_stream():
                # 在生成器内部加锁，确保模型在流式生成全过程被独占
                async with model_lock:
                    # 设置随机种子 (必须在锁内，因为是全局状态)
                    if request.seed >= 0:
                        torch.manual_seed(request.seed)
                        torch.cuda.manual_seed(request.seed)

                    sr = model.tts_model.sample_rate
                    print(f"Starting stream | SR: {sr}")
                    
                    # 如果是 audio 模式且要求 wav，先发送一个极其巨大的 WAV Header 模拟流式
                    if request.stream_format == "audio" and request.response_format == "wav":
                        yield create_wav_header(sr, 1, 16)

                    try:
                        for idx, chunk in enumerate(model.generate_streaming(
                            text=request.input,
                            prompt_text=prompt_text,
                            prompt_wav_path=prompt_wav_path,
                            reference_wav_path=ref_path,
                            cfg_value=request.cfg_value,
                            inference_timesteps=request.inference_timesteps,
                        )):
                            # 归一化
                            max_abs = np.abs(chunk).max()
                            if max_abs > 1.0:
                                chunk = chunk / (max_abs + 1e-5)
                            
                            if idx % 20 == 0:
                                 print(f"Chunk {idx}...")
                            
                            # 标准化转换：统一转为 PCM16
                            pcm16_data = (chunk * 32767).astype(np.int16).tobytes()

                            if request.stream_format == "sse":
                                b64_chunk = base64.b64encode(pcm16_data).decode('utf-8')
                                yield f"data: {json.dumps({'audio': b64_chunk, 'sample_rate': sr})}\n\n"
                            else:
                                yield pcm16_data
                        
                        if request.stream_format == "sse":
                            yield "data: [DONE]\n\n"
                    except Exception as e:
                        import traceback
                        print("Streaming Error Traceback:")
                        traceback.print_exc()
                        print(f"Streaming Error: {e}")
                    finally:
                        # 清理临时文件
                        for f in temp_files:
                            if f and os.path.exists(f):
                                try: os.remove(f)
                                except: pass

            return StreamingResponse(
                generate_stream(), 
                media_type="text/event-stream" if request.stream_format == "sse" else f"audio/{request.response_format}"
            )

        # --- 非流式逻辑 ---
        else:
            async with model_lock:
                # 设置随机种子
                if request.seed >= 0:
                    torch.manual_seed(request.seed)
                    torch.cuda.manual_seed(request.seed)

                wav = model.generate(
                    text=request.input,
                    prompt_text=prompt_text,
                    prompt_wav_path=prompt_wav_path,
                    reference_wav_path=ref_path,
                    cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps,
                )
            
            # 清理临时文件
            for f in temp_files:
                if f and os.path.exists(f):
                    try: os.remove(f)
                    except: pass

            buffer = io.BytesIO()
            sf.write(buffer, wav, model.tts_model.sample_rate, format=request.response_format.upper())
            buffer.seek(0)
            return StreamingResponse(buffer, media_type=f"audio/{request.response_format}")

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 发生异常也要清理
        for f in temp_files:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
