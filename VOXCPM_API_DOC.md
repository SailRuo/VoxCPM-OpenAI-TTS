# VoxCPM2 All-In-One API 文档

本文档描述了 VoxCPM2 语音合成服务的 API 接口协议。该接口兼容流式与非流式调用，支持多种合成模式（零样本、语音设计、音色克隆、情感延续、极致克隆）。

## 基础信息

- **服务地址**: `http://localhost:8001`
- **接口路径**: `/v1/audio/speech`
- **请求方法**: `POST`
- **内容类型**: `application/json`

---

## 请求参数 (Request Body)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **input** | string | (必填) | 目标转换内容文本。若使用“语音设计”模式，需在开头包含括号，如 `(A happy girl)你好`。 |
| **model** | string | "voxcpm-2" | 模型名称标识。 |
| **voice** | string | "default" | 身份参考音色名称（对应服务器 `refvoice/` 目录下的文件名）。 |
| **voice_data** | string | "" | 身份参考音频的 **Base64** 编码数据（若提供，将优先于 `voice` 参数）。 |
| **prompt_text** | string | "" | 上下文参考文本（情感延续/极速克隆模式必填）。 |
| **prompt_voice** | string | "default" | 风格内容参考音频名称。 |
| **prompt_voice_data** | string | "" | 风格内容参考音频的 **Base64** 编码数据。 |
| **stream** | boolean | true | 是否启用流式输出。 |
| **stream_format** | string | "audio" | 流式格式可选：`audio` (原始字节流) 或 `sse` (Server-Sent Events JSON)。 |
| **response_format** | string | "wav" | 响应音频格式（支持 wav, mp3, flac 等）。 |
| **speed** | float | 1.0 | 语速调整（目前模型主要由 Prompt 控制）。 |
| **cfg_value** | float | 2.0 | Classifier-Free Guidance 权重（建议 1.5 - 3.0）。 |
| **inference_timesteps** | integer | 10 | 推理步数（值越大质量越好但延迟越高，流式推荐 10-15）。 |
| **seed** | integer | -1 | 随机种子。固定种子可复现语气，-1 为随机。 |

---

## 响应说明

### 1. 非流式响应 (`stream: false`)
直接返回二进制音频数据流。
- **Content-Type**: `audio/wav` (根据 `response_format` 改变)

### 2. SSE 流式响应 (`stream: true`, `stream_format: "sse"`)
返回一系列 `data: ` 格式的消息。
- **消息内容**: `{"audio": "BASE64_DATA", "sample_rate": 48000}`
- **结束标识**: `data: [DONE]`

---

## 合成模式详细示例

### 1. 零样本合成 (Zero-shot)
随机生成一个新音色。
```json
{
  "input": "你好，这是零样本随机合成测试。",
  "stream": true,
  "stream_format": "sse"
}
```

### 2. 语音设计 (Voice Design)
通过文字描述捏取音色。
```json
{
  "input": "(A mature man with a heavy gravelly voice, speaking fast) 欢迎来到语音设计模式。",
  "stream": false
}
```

### 3. 音色克隆 (Voice Cloning)
根据已有的音色文件进行克隆。
```json
{
  "input": "正在使用紫风的音色为您朗读。",
  "voice": "zifeng"
}
```

### 4. 情感延续 (Continuation)
延续参考音频的情感、语调和背景环境。
```json
{
  "input": "这是背景情感延续的内容。",
  "prompt_voice": "zifeng",
  "prompt_text": "我是参考音频里的原话内容。",
  "stream": true
}
```

### 5. 极致克隆 / 风格迁移 (Ultimate Cloning)
用 A 的声音去模仿 B 的情感细节。
```json
{
  "input": "这是极致克隆测试，音色来自 A，情感来自 B。",
  "voice": "zifeng",
  "prompt_voice": "doubao",
  "prompt_text": "刚才豆包说话的情感参考。",
  "stream": true
}
```

---

## 错误码说明

| 状态码 | 说明 | 原因 |
| :--- | :--- | :--- |
| **400** | Bad Request | 参数缺失（如情感延续模式未传 Prompt）或参考文件不存在。 |
| **500** | Server Error | 模型未加载或推理过程中崩溃。 |

---

## 调用建议

1. **并发控制**: 后端已加锁保护。并发请求时，后续请求会进入队列排队，请确保客户端有足够的超时等待时间。
2. **Base64 上传**: 在 Web 应用中，建议直接通过 `voice_data` 传输用户录制的 Base64，省去文件上传服务器的步骤。
3. **流式体验**: 推荐使用 `stream: true` 和 `sse` 格式，可以实现边生成边播放，大幅降低首字延迟。

---

## Python 调用示例 (使用 OpenAI 官方 SDK)

本项目 API 完全兼容 OpenAI TTS 协议，你可以直接使用官方 `openai` 库：

```python
from openai import OpenAI

client = OpenAI(
    api_key="none", # 随便填
    base_url="http://localhost:8081/v1"
)

# 标准调用
response = client.audio.speech.create(
    model="voxcpm-2",
    voice="default",
    input="你好，这是一段通过 OpenAI SDK 调用的语音。",
    response_format="wav"
)

response.stream_to_file("output.wav")
```
