# VoxCPM-API

本项目是为 **VoxCPM2** 模型封装的 **OpenAI 兼容接口服务**。它可以让你像使用 OpenAI TTS 一样，快速调用 VoxCPM2 这款强大的语音合成模型。

**VoxCPM2** 模型本身是一个无分词器（Tokenizer-free）、扩散自回归（Diffusion Autoregressive）架构的深度学习模型，拥有 **20 亿参数**，支持 **30 种语言**，能够输出 **48kHz** 高保真音质。

[![GitHub](https://img.shields.io/badge/GitHub-VoxCPM-blue?logo=github)](https://github.com/OpenBMB/VoxCPM)
[![Docs](https://img.shields.io/badge/Docs-ReadTheDocs-8CA1AF)](https://voxcpm.readthedocs.io/en/latest/)
[![Demo](https://img.shields.io/badge/Live%20Playground-Demo-orange)](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo)
[![Audio Samples](https://img.shields.io/badge/Audio%20Samples-Demo%20Page-green)](https://openbmb.github.io/voxcpm2-demopage)
[![Discord](https://img.shields.io/badge/Discord-VoxCPM-5865F2?logo=discord&logoColor=white)](https://discord.gg/KZUx7tVNwz)
[![Lark](https://img.shields.io/badge/飞书群-VoxCPM-00D6B9?logo=lark&logoColor=white)](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=acds0b9d-23d8-4d7e-b696-d200f3e22a7f)

## 核心亮点

- 🌍 **支持 30 种语言** — 无需语言标签，直接输入任何支持的语言文本即可。
- 🎨 **语音设计 (Voice Design)** — 仅通过自然语言描述（性别、年龄、音调、情感、语速等）即可生成全新的音色，无需参考音频。
- 🎛️ **可控音色克隆 (Controllable Cloning)** — 仅需一段短音频即可克隆音色，并支持通过风格引导微调情感、节奏和表现力。
- 🎙️ **极致克隆 (Ultimate Cloning)** — 提供参考音频及其准确文本，完美复刻原声的每一个细微特征。
- 🔊 **48kHz 录音室级输出** — 输入 16kHz 参考音频，通过内置的 AudioVAE V2 自动超分辨率输出 48kHz 音频。
- 🧠 **上下文感知合成** — 自动根据文本内容推断合适的停顿、语气和情感表现。
- ⚡ **实时流式推理** — 在 RTX 4090 上 RTF 低至 ~0.3，通过 [Nano-VLLM](https://github.com/a710128/nanovllm-voxcpm) 加速可达 ~0.13。
- 📜 **完全开源 & 商业友好** — 采用 Apache-2.0 协议，免费商用。


<summary><b>支持的语言 (30)</b></summary>

阿拉伯语、缅甸语、中文、丹麦语、荷兰语、英语、芬兰语、法语、德语、希腊语、希伯来语、印地语、印度尼西亚语、意大利语、日语、高棉语、韩语、老挝语、马来语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、斯瓦希里语、瑞典语、塔加洛语、泰语、土耳其语、越南语。

**中国方言：** 四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话。

---

## 快速开始

### 安装

```bash
pip install voxcpm
```

**运行环境:** Python ≥ 3.10, PyTorch ≥ 2.5.0, CUDA ≥ 12.0 · [查看完整指南 →](https://voxcpm.readthedocs.io/en/latest/quickstart.html)

### 基础文本转语音

```python
from voxcpm import VoxCPM
import soundfile as sf

model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

wav = model.generate(
    text="VoxCPM2 带来了多语言支持、创意语音设计和可控音色克隆。",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("output.wav", wav, model.tts_model.sample_rate)
```

### 语音设计 (Voice Design)

在 `text` 开头使用括号描述声音特征，后接合成内容：

```python
wav = model.generate(
    text="(一位年轻女性，声音温柔甜美)欢迎使用 VoxCPM2！",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("voice_design.wav", wav, model.tts_model.sample_rate)
```

### 实时流式生成

```python
import numpy as np

chunks = []
for chunk in model.generate_streaming(text="使用 VoxCPM 进行流式合成非常简单！"):
    chunks.append(chunk)
wav = np.concatenate(chunks)
sf.write("streaming.wav", wav, model.tts_model.sample_rate)
```

---

## 🚀 生产级部署 (Web 控制台 & API)

本项目自带了一个完全兼容 **OpenAI TTS** 标准的 API 服务，方便各种第三方客户端直接接入。

### 1. 使用 Docker 启动 (推荐)
```bash
docker compose up -d --build
```

### 2. 访问语音工作室
通过浏览器访问： 👉 **[http://localhost:8081](http://localhost:8081)**

### 3. API 集成 (兼容 OpenAI)
- **Base URL**: `http://localhost:8081`
- **接口路径**: `/v1/audio/speech`
- **调用建议**: 本服务已实现并发排队锁，确保在高并发请求下模型推理不崩溃。详情请参考 **[API 接口文档](./VOXCPM_API_DOC.md)**。

---

## 模型详情

| 属性 | 参数值 |
|---|---|
| 架构 | 无分词器扩散自回归 (LocEnc → TSLM → RALM → LocDiT) |
| 骨干网络 | 基于 MiniCPM-4，总参数量 2B |
| Audio VAE | AudioVAE V2 (不对称编解码，16kHz 入 -> 48kHz 出) |
| 训练数据 | 200万+ 小时多语言语音 |
| LM Token 速率 | 6.25 Hz |
| 最大序列长度 | 8192 tokens |
| 数据精度 | bfloat16 |
| 显存占用 | ~8 GB |
| RTF (RTX 4090) | ~0.30 (标准) / ~0.13 (Nano-vLLM 加速) |

## 局限性

- **随机性**：语音设计和风格控制的结果因运行而异，建议生成 1-3 次以获得理想效果。
- **语言差异**：性能因各语言训练数据的多少而有所差异。
- **超长文本**：处理极长或情感极其剧烈的输入时可能存在不稳定性。
- **伦理声明**：严禁将本模型用于诈骗、冒充或传播虚假信息。AI 生成内容应明确标注。

## 微调指南

可以通过短短 5-10 分钟的学习数据进行全量或 LoRA 微调：

```bash
# LoRA 微调 (推荐)
python scripts/train_voxcpm_finetune.py \
    --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml
```

## 引用

```bibtex
@article{voxcpm2_2026,
  title   = {VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning},
  author  = {VoxCPM Team},
  journal = {GitHub},
  year    = {2026},
}
```

## 许可证

本项目基于 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议发布，免费商用。
