# TingWu 语音转写服务实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个基于 FunASR + CapsWriter-Offline 优化的中文语音转写服务，支持文件上传转写、实时流式转写、说话人识别（甲/乙标注），并通过 Docker 部署。

**Architecture:**
- **核心层**: FunASR (Paraformer-large + VAD + Punctuation + Speaker) 提供基础 ASR 能力
- **优化层**: CapsWriter-Offline 的热词纠错、音素 RAG、LLM 润色等技术提升准确率
- **服务层**: FastAPI HTTP 接口（文件上传）+ WebSocket 接口（实时流式）
- **部署层**: Docker Compose 编排，支持 CPU/GPU 模式

**Tech Stack:** Python 3.10+, FastAPI, WebSocket, FunASR, PyTorch, Docker, Redis (可选队列), Nginx (可选反代)

---

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TingWu Speech Service                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   HTTP API      │    │  WebSocket API  │    │   Admin API     │     │
│  │  /transcribe    │    │  /ws/realtime   │    │  /health        │     │
│  │  (文件上传)      │    │  (实时流式)      │    │  /hotwords      │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           └──────────────────────┼──────────────────────┘               │
│                                  ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Transcription Engine                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │ ASR Module  │  │ SPK Module  │  │ Optimizer   │                │ │
│  │  │ (FunASR)    │  │ (CAMPPlus)  │  │ (CapsWriter)│                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                  │                                      │
│  ┌───────────────────────────────┴───────────────────────────────────┐ │
│  │                         Model Layer                                │ │
│  │  • Paraformer-large (ASR)      • FSMN-VAD (端点检测)               │ │
│  │  • Paraformer-streaming (实时)  • CT-Transformer (标点)            │ │
│  │  • CAMPPlus (说话人embedding)   • ClusterBackend (说话人聚类)      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 项目基础结构搭建

### Task 1: 初始化项目目录结构

**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/main.py`
- Create: `requirements.txt`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`

**Step 1: 创建项目目录结构**

```bash
mkdir -p src/{api,core,models,utils,services}
mkdir -p configs data/{models,hotwords,uploads,outputs} tests
touch src/__init__.py src/api/__init__.py src/core/__init__.py
touch src/models/__init__.py src/utils/__init__.py src/services/__init__.py
```

**Step 2: 创建配置文件 `src/config.py`**

```python
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, Literal

class Settings(BaseSettings):
    """应用配置"""
    # 服务配置
    app_name: str = "TingWu Speech Service"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # 路径配置
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = data_dir / "models"
    hotwords_dir: Path = data_dir / "hotwords"
    uploads_dir: Path = data_dir / "uploads"
    outputs_dir: Path = data_dir / "outputs"

    # FunASR 模型配置
    asr_model: str = "paraformer-zh"
    asr_model_online: str = "paraformer-zh-streaming"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc-c"
    spk_model: str = "cam++"

    # 设备配置
    device: Literal["cuda", "cpu"] = "cuda"
    ngpu: int = 1
    ncpu: int = 4

    # 热词配置
    hotwords_file: str = "hotwords.txt"
    hotwords_threshold: float = 0.85

    # LLM 优化配置 (可选)
    llm_enable: bool = False
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"

    # WebSocket 配置
    ws_chunk_size: int = 9600  # 600ms @ 16kHz
    ws_chunk_interval: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# 确保目录存在
for dir_path in [settings.data_dir, settings.models_dir, settings.hotwords_dir,
                 settings.uploads_dir, settings.outputs_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
```

**Step 3: 创建 requirements.txt**

```text
# Core
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
websockets>=12.0
python-multipart>=0.0.6
pydantic>=2.5.0
pydantic-settings>=2.1.0

# FunASR
funasr>=1.1.0
torch>=2.0.0
torchaudio>=2.0.0
modelscope>=1.10.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.1
ffmpeg-python>=0.2.0
numpy>=1.24.0

# Hot word optimization (from CapsWriter)
pypinyin>=0.50.0
numba>=0.58.0
jieba>=0.42.1

# LLM (optional)
requests>=2.31.0
httpx>=0.26.0

# Utils
aiofiles>=23.2.0
python-dotenv>=1.0.0
loguru>=0.7.0
rich>=13.7.0
```

**Step 4: 创建 Dockerfile**

```dockerfile
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 5: 创建 docker-compose.yml**

```yaml
version: '3.8'

services:
  tingwu:
    build:
      context: .
      dockerfile: Dockerfile
    image: tingwu-speech-service:latest
    container_name: tingwu-service
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      # 挂载模型缓存 (避免每次重新下载)
      - model-cache:/root/.cache/modelscope
    environment:
      - DEVICE=cuda
      - NGPU=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # CPU 版本 (备选)
  tingwu-cpu:
    build:
      context: .
      dockerfile: Dockerfile
    image: tingwu-speech-service:cpu
    container_name: tingwu-service-cpu
    ports:
      - "8001:8000"
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
      - model-cache:/root/.cache/modelscope
    environment:
      - DEVICE=cpu
      - NGPU=0
    profiles:
      - cpu
    restart: unless-stopped

volumes:
  model-cache:
```

**Step 6: 创建 .env.example**

```bash
# 设备配置
DEVICE=cuda
NGPU=1
NCPU=4

# 服务配置
HOST=0.0.0.0
PORT=8000
DEBUG=false

# 热词配置
HOTWORDS_THRESHOLD=0.85

# LLM 配置 (可选)
LLM_ENABLE=false
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434
```

**Step 7: 运行目录创建脚本验证结构**

Run: `tree -L 3 /data/temp41/tingwu/`
Expected: 显示完整的项目目录结构

**Step 8: Commit**

```bash
git init
git add .
git commit -m "$(cat <<'EOF'
chore: initialize project structure

- Add src/ directory with config and main entry
- Add requirements.txt with FunASR and FastAPI dependencies
- Add Dockerfile for containerization
- Add docker-compose.yml for GPU/CPU deployment
- Add .env.example for configuration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 实现 FunASR 模型加载器

**Files:**
- Create: `src/models/asr_loader.py`
- Create: `src/models/model_manager.py`
- Test: `tests/test_model_loader.py`

**Step 1: 编写模型加载器测试**

```python
# tests/test_model_loader.py
import pytest
from src.models.asr_loader import ASRModelLoader

def test_model_loader_initialization():
    """测试模型加载器可以正确初始化"""
    loader = ASRModelLoader(device="cpu", ngpu=0)
    assert loader is not None
    assert loader.device == "cpu"

def test_model_loader_lazy_load():
    """测试模型懒加载机制"""
    loader = ASRModelLoader(device="cpu", ngpu=0)
    # 模型应该还未加载
    assert loader._asr_model is None
    # 访问属性时才加载
    model = loader.asr_model
    assert model is not None
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_model_loader.py -v`
Expected: FAIL - 模块不存在

**Step 3: 实现 ASRModelLoader**

```python
# src/models/asr_loader.py
"""FunASR 模型加载器 - 懒加载模式"""
import logging
from typing import Optional, Dict, Any
from funasr import AutoModel

logger = logging.getLogger(__name__)

class ASRModelLoader:
    """ASR 模型加载器，支持懒加载和模型组合"""

    def __init__(
        self,
        device: str = "cuda",
        ngpu: int = 1,
        ncpu: int = 4,
        asr_model: str = "paraformer-zh",
        asr_model_online: str = "paraformer-zh-streaming",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc-c",
        spk_model: Optional[str] = "cam++",
    ):
        self.device = device
        self.ngpu = ngpu
        self.ncpu = ncpu

        # 模型名称配置
        self._asr_model_name = asr_model
        self._asr_model_online_name = asr_model_online
        self._vad_model_name = vad_model
        self._punc_model_name = punc_model
        self._spk_model_name = spk_model

        # 懒加载的模型实例
        self._asr_model: Optional[AutoModel] = None
        self._asr_model_online: Optional[AutoModel] = None
        self._asr_model_with_spk: Optional[AutoModel] = None

    @property
    def asr_model(self) -> AutoModel:
        """获取离线 ASR 模型 (VAD + ASR + Punc)"""
        if self._asr_model is None:
            logger.info(f"Loading offline ASR model: {self._asr_model_name}")
            self._asr_model = AutoModel(
                model=self._asr_model_name,
                vad_model=self._vad_model_name,
                punc_model=self._punc_model_name,
                device=self.device,
                ngpu=self.ngpu,
                ncpu=self.ncpu,
                disable_pbar=True,
                disable_log=True,
            )
            logger.info("Offline ASR model loaded successfully")
        return self._asr_model

    @property
    def asr_model_online(self) -> AutoModel:
        """获取在线流式 ASR 模型"""
        if self._asr_model_online is None:
            logger.info(f"Loading online ASR model: {self._asr_model_online_name}")
            self._asr_model_online = AutoModel(
                model=self._asr_model_online_name,
                device=self.device,
                ngpu=self.ngpu,
                ncpu=self.ncpu,
                disable_pbar=True,
                disable_log=True,
            )
            logger.info("Online ASR model loaded successfully")
        return self._asr_model_online

    @property
    def asr_model_with_spk(self) -> AutoModel:
        """获取带说话人识别的 ASR 模型"""
        if self._asr_model_with_spk is None:
            logger.info("Loading ASR model with speaker diarization")
            self._asr_model_with_spk = AutoModel(
                model=self._asr_model_name,
                vad_model=self._vad_model_name,
                punc_model=self._punc_model_name,
                spk_model=self._spk_model_name,
                device=self.device,
                ngpu=self.ngpu,
                ncpu=self.ncpu,
                disable_pbar=True,
                disable_log=True,
            )
            logger.info("ASR model with speaker loaded successfully")
        return self._asr_model_with_spk

    def transcribe(
        self,
        audio_input,
        hotwords: Optional[str] = None,
        with_speaker: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """执行转写"""
        params = {
            "input": audio_input,
            "sentence_timestamp": True,
            "batch_size_s": 300,
        }
        if hotwords:
            params["hotword"] = hotwords
        params.update(kwargs)

        model = self.asr_model_with_spk if with_speaker else self.asr_model
        result = model.generate(**params)
        return result[0] if result else {}
```

**Step 4: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_model_loader.py -v`
Expected: PASS

**Step 5: 创建模型管理器单例**

```python
# src/models/model_manager.py
"""全局模型管理器"""
from typing import Optional
from src.config import settings
from src.models.asr_loader import ASRModelLoader

class ModelManager:
    """模型管理器单例"""
    _instance: Optional['ModelManager'] = None
    _loader: Optional[ASRModelLoader] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def loader(self) -> ASRModelLoader:
        if self._loader is None:
            self._loader = ASRModelLoader(
                device=settings.device,
                ngpu=settings.ngpu,
                ncpu=settings.ncpu,
                asr_model=settings.asr_model,
                asr_model_online=settings.asr_model_online,
                vad_model=settings.vad_model,
                punc_model=settings.punc_model,
                spk_model=settings.spk_model,
            )
        return self._loader

    def preload_models(self, with_speaker: bool = True):
        """预加载模型"""
        _ = self.loader.asr_model
        _ = self.loader.asr_model_online
        if with_speaker:
            _ = self.loader.asr_model_with_spk

model_manager = ModelManager()
```

**Step 6: Commit**

```bash
git add src/models/ tests/
git commit -m "$(cat <<'EOF'
feat: add FunASR model loader with lazy loading

- Implement ASRModelLoader with lazy model initialization
- Support offline/online/speaker models
- Add ModelManager singleton for global access
- Add unit tests for model loader

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 实现热词纠错模块 (基于 CapsWriter)

**Files:**
- Create: `src/core/hotword/__init__.py`
- Create: `src/core/hotword/phoneme.py`
- Create: `src/core/hotword/corrector.py`
- Create: `src/core/hotword/rag.py`
- Test: `tests/test_hotword.py`

**Step 1: 编写热词纠错测试**

```python
# tests/test_hotword.py
import pytest
from src.core.hotword.corrector import PhonemeCorrector

@pytest.fixture
def corrector():
    c = PhonemeCorrector(threshold=0.8, similar_threshold=0.6)
    c.update_hotwords("Claude\nBilibili\n麦当劳\n肯德基")
    return c

def test_chinese_correction(corrector):
    """测试中文热词纠错"""
    result = corrector.correct("我想去吃买当劳")
    assert "麦当劳" in result.text

def test_english_correction(corrector):
    """测试英文热词纠错"""
    result = corrector.correct("Hello klaude")
    assert "Claude" in result.text

def test_similar_phoneme_matching(corrector):
    """测试相似音素匹配"""
    result = corrector.correct("肯得鸡很好吃")
    assert "肯德基" in result.text
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_hotword.py -v`
Expected: FAIL - 模块不存在

**Step 3: 实现音素处理模块**

```python
# src/core/hotword/phoneme.py
"""音素处理模块 - 基于 CapsWriter-Offline"""
import re
from dataclasses import dataclass
from typing import List, Literal, Tuple

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False


@dataclass(frozen=True, slots=True)
class Phoneme:
    """音素数据类"""
    value: str
    lang: Literal['zh', 'en', 'num', 'other']
    is_word_start: bool = False
    is_word_end: bool = False
    char_start: int = 0
    char_end: int = 0

    @property
    def is_tone(self) -> bool:
        return self.value.isdigit()

    @property
    def info(self) -> Tuple:
        return (self.value, self.lang, self.is_word_start,
                self.is_word_end, self.is_tone, self.char_start, self.char_end)


# 相似音素集合 (用于模糊匹配)
SIMILAR_PHONEMES = [
    {'an', 'ang'}, {'en', 'eng'}, {'in', 'ing'},
    {'ian', 'iang'}, {'uan', 'uang'},
    {'z', 'zh'}, {'c', 'ch'}, {'s', 'sh'},
    {'l', 'n'}, {'f', 'h'}, {'ai', 'ei'}
]


def get_phoneme_info(text: str, split_char: bool = True) -> List[Phoneme]:
    """提取文本的音素序列"""
    if not HAS_PYPINYIN:
        return [Phoneme(c, 'zh', char_start=i, char_end=i+1)
                for i, c in enumerate(text)]

    seq = []
    pos = 0

    while pos < len(text):
        c = text[pos]

        # 处理中文字符
        if '\u4e00' <= c <= '\u9fff':
            start = pos
            pos += 1
            while pos < len(text) and '\u4e00' <= text[pos] <= '\u9fff':
                pos += 1
            frag = text[start:pos]

            try:
                pi = pinyin(frag, style=Style.INITIALS, strict=False)
                pf = pinyin(frag, style=Style.FINALS, strict=False)
                pt = pinyin(frag, style=Style.TONE3, neutral_tone_with_five=True)

                for i in range(min(len(frag), len(pi), len(pf), len(pt))):
                    idx = start + i
                    init, fin, tone = pi[i][0], pf[i][0], pt[i][0]

                    if init:
                        seq.append(Phoneme(init, 'zh', is_word_start=True,
                                          char_start=idx, char_end=idx+1))
                    if fin:
                        seq.append(Phoneme(fin, 'zh', is_word_start=not init,
                                          char_start=idx, char_end=idx+1))
                    if tone and tone[-1].isdigit():
                        seq.append(Phoneme(tone[-1], 'zh', is_word_end=True,
                                          char_start=idx, char_end=idx+1))
            except Exception:
                for i, char in enumerate(frag):
                    seq.append(Phoneme(char, 'zh', is_word_start=True,
                                      is_word_end=True, char_start=start+i,
                                      char_end=start+i+1))

        # 处理英文/数字
        elif 'a' <= c.lower() <= 'z' or '0' <= c <= '9':
            start = pos
            pos += 1
            while pos < len(text):
                cur = text[pos]
                if not ('a' <= cur.lower() <= 'z' or '0' <= cur <= '9'):
                    break
                if (text[pos-1].islower() and cur.isupper()) or \
                   (text[pos-1].isalpha() and cur.isdigit()) or \
                   (text[pos-1].isdigit() and cur.isalpha()):
                    break
                pos += 1

            token = text[start:pos].lower()
            lang = 'num' if token.isdigit() else 'en'

            if split_char:
                for i, char in enumerate(token):
                    seq.append(Phoneme(char, lang, is_word_start=(i==0),
                                      is_word_end=(i==len(token)-1),
                                      char_start=start+i, char_end=start+i+1))
            else:
                seq.append(Phoneme(token, lang, is_word_start=True,
                                  is_word_end=True, char_start=start, char_end=pos))
        else:
            pos += 1

    return seq


def get_phoneme_cost(p1: Phoneme, p2: Phoneme) -> float:
    """计算两个音素之间的距离代价"""
    if p1.lang != p2.lang:
        return 1.0
    if p1.value == p2.value:
        return 0.0

    # 中文相似音素
    if p1.lang == 'zh':
        pair = {p1.value, p2.value}
        for s in SIMILAR_PHONEMES:
            if pair.issubset(s):
                return 0.5

    # 英文 LCS 相似度
    if p1.lang == 'en':
        lcs = _lcs_length(p1.value, p2.value)
        max_len = max(len(p1.value), len(p2.value))
        return 1.0 - (lcs / max_len) if max_len > 0 else 0.0

    return 1.0


def _lcs_length(s1: str, s2: str) -> int:
    """计算最长公共子序列长度"""
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    if n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]
```

**Step 4: 实现 FastRAG 检索**

```python
# src/core/hotword/rag.py
"""快速 RAG 检索模块"""
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from src.core.hotword.phoneme import Phoneme


if HAS_NUMBA and HAS_NUMPY:
    @njit(cache=True)
    def _fuzzy_substring_numba(main, sub):
        """Numba 加速的模糊子串距离计算"""
        n, m = len(sub), len(main)
        if n == 0 or m == 0:
            return float(n)

        dp = np.zeros((n+1, m+1), dtype=np.float32)
        for i in range(1, n+1):
            dp[i, 0] = float(i)

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0.0 if sub[i-1] == main[j-1] else 1.0
                dp[i, j] = min(dp[i-1, j] + 1.0,
                              dp[i, j-1] + 1.0,
                              dp[i-1, j-1] + cost)

        return np.min(dp[n, 1:])


class FastRAG:
    """快速 RAG 热词检索"""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.ph_to_id: Dict[str, int] = {}
        self.index: Dict[int, List[Tuple[str, any]]] = defaultdict(list)
        self.hotword_count = 0

    def _encode(self, phs: List[Phoneme]):
        """将音素序列编码为整数序列"""
        ids = []
        for p in phs:
            if p.value not in self.ph_to_id:
                self.ph_to_id[p.value] = len(self.ph_to_id) + 1
            ids.append(self.ph_to_id[p.value])

        if HAS_NUMPY:
            return np.array(ids, dtype=np.int32)
        return ids

    def add_hotwords(self, hotwords: Dict[str, List[Phoneme]]):
        """添加热词到索引"""
        for hw, phs in hotwords.items():
            if not phs:
                continue
            codes = self._encode(phs)
            # 使用前两个音素建立倒排索引
            for i in range(min(len(codes), 2)):
                self.index[codes[i]].append((hw, codes))
            self.hotword_count += 1

    def search(self, input_phs: List[Phoneme], top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索匹配的热词"""
        if not input_phs:
            return []

        input_codes = self._encode(input_phs)
        unique = set(input_codes.tolist() if HAS_NUMPY else input_codes)

        # 收集候选
        candidates = []
        for c in unique:
            candidates.extend(self.index.get(c, []))

        seen = set()
        results = []

        for hw, cands in candidates:
            if hw in seen or len(cands) > len(input_codes) + 3:
                continue
            seen.add(hw)

            # 计算距离
            if HAS_NUMBA and HAS_NUMPY:
                dist = _fuzzy_substring_numba(input_codes, cands)
            else:
                dist = self._python_dist(input_codes, cands)

            score = 1.0 - (dist / len(cands))
            if score >= self.threshold:
                results.append((hw, round(float(score), 3)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _python_dist(self, main, sub) -> float:
        """纯 Python 实现的模糊子串距离"""
        n, m = len(sub), len(main)
        dp = [[0.0] * (m+1) for _ in range(n+1)]

        for i in range(1, n+1):
            dp[i][0] = float(i)

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0.0 if sub[i-1] == main[j-1] else 1.0
                dp[i][j] = min(dp[i-1][j] + 1.0,
                              dp[i][j-1] + 1.0,
                              dp[i-1][j-1] + cost)

        return min(dp[n][1:]) if m > 0 else float(n)
```

**Step 5: 实现热词纠错器**

```python
# src/core/hotword/corrector.py
"""热词纠错器"""
import os
import threading
from typing import List, Dict, Tuple, NamedTuple, Optional

from src.core.hotword.phoneme import Phoneme, get_phoneme_info, get_phoneme_cost
from src.core.hotword.rag import FastRAG


class MatchResult(NamedTuple):
    start: int
    end: int
    score: float
    hotword: str


class CorrectionResult(NamedTuple):
    text: str
    matches: List[Tuple[str, float]]
    similars: List[Tuple[str, float]]


class PhonemeCorrector:
    """基于音素的热词纠错器"""

    def __init__(self, threshold: float = 0.8, similar_threshold: float = None):
        self.threshold = threshold
        self.similar_threshold = similar_threshold or (threshold - 0.2)
        self.hotwords: Dict[str, List[Phoneme]] = {}
        self.fast_rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.1)
        self._lock = threading.Lock()

    def update_hotwords(self, text: str) -> int:
        """从文本更新热词"""
        lines = [l.strip() for l in text.splitlines()
                 if l.strip() and not l.strip().startswith('#')]

        new_hotwords = {}
        for hw in lines:
            phs = get_phoneme_info(hw)
            if phs:
                new_hotwords[hw] = phs

        with self._lock:
            self.hotwords = new_hotwords
            self.fast_rag = FastRAG(threshold=min(self.threshold, self.similar_threshold) - 0.1)
            self.fast_rag.add_hotwords(new_hotwords)

        return len(new_hotwords)

    def load_hotwords_file(self, path: str) -> int:
        """从文件加载热词"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return self.update_hotwords(f.read())
        return 0

    def correct(self, text: str, top_k: int = 10) -> CorrectionResult:
        """执行热词纠错"""
        input_phs = get_phoneme_info(text)
        if not input_phs:
            return CorrectionResult(text, [], [])

        with self._lock:
            fast_results = self.fast_rag.search(input_phs, top_k=100)
            processed = [p.info for p in input_phs]
            matches, similars = self._find_matches(fast_results, processed)

        new_text, final_matches, _ = self._resolve_and_replace(text, matches)
        return CorrectionResult(
            new_text,
            final_matches,
            [(m.hotword, m.score) for m in similars[:top_k]]
        )

    def _find_matches(self, fast_results, input_processed):
        """查找匹配的热词"""
        matches = []
        similars = []
        input_len = len(input_processed)

        for hw, _ in fast_results:
            hw_phs = self.hotwords[hw]
            hw_compare = [p.info[:5] for p in hw_phs]
            target_len = len(hw_compare)

            if target_len > input_len:
                continue

            for i in range(input_len - target_len + 1):
                seg = input_processed[i:i + target_len]

                # 检查词边界
                if seg[0][1] != 'en' and seg[0][0] != hw_compare[0][0]:
                    continue
                if not seg[0][2]:  # is_word_start
                    continue

                is_end_ok = seg[-1][3] or (
                    i + target_len < input_len and
                    input_processed[i + target_len][1] == 'zh' and
                    input_processed[i + target_len][4]
                )
                if not is_end_ok:
                    continue

                score = self._fuzzy_score(hw_compare, seg)
                m = MatchResult(seg[0][5], seg[-1][6], score, hw)
                similars.append(m)

                if score >= self.threshold:
                    matches.append(m)

        seen = set()
        sorted_sims = sorted(similars, key=lambda x: x.score, reverse=True)
        sims_final = [m for m in sorted_sims
                      if m.score >= self.similar_threshold
                      and not (m.hotword in seen or seen.add(m.hotword))]

        return matches, sims_final

    def _fuzzy_score(self, target, source) -> float:
        """计算模糊匹配得分"""
        n = len(target)
        if n == 0:
            return 0.0

        total_cost = 0.0
        for t, s in zip(target, source):
            if t[0] != s[0]:  # value 不同
                if t[1] == s[1] == 'zh':  # 都是中文
                    # 检查相似音素
                    from src.core.hotword.phoneme import SIMILAR_PHONEMES
                    pair = {t[0], s[0]}
                    found = False
                    for sim_set in SIMILAR_PHONEMES:
                        if pair.issubset(sim_set):
                            total_cost += 0.5
                            found = True
                            break
                    if not found:
                        total_cost += 1.0
                else:
                    total_cost += 1.0

        return 1.0 - (total_cost / n)

    def _resolve_and_replace(self, text, matches):
        """解决冲突并替换文本"""
        matches.sort(key=lambda x: (x.score, x.end - x.start), reverse=True)

        final = []
        all_info = []
        occupied = []
        seen = set()

        for m in matches:
            if (m.hotword, m.score) not in seen:
                all_info.append((m.hotword, m.score))
                seen.add((m.hotword, m.score))

            if m.score < self.threshold:
                continue

            # 检查是否与已占用区间重叠
            overlaps = any(not (m.end <= rs or m.start >= re) for rs, re in occupied)
            if overlaps:
                continue

            if text[m.start:m.end] != m.hotword:
                final.append(m)
            occupied.append((m.start, m.end))

        # 执行替换
        result = list(text)
        final.sort(key=lambda x: x.start, reverse=True)
        for m in final:
            result[m.start:m.end] = list(m.hotword)

        return "".join(result), [(m.hotword, m.score) for m in sorted(final, key=lambda x: x.start)], all_info
```

**Step 6: 创建模块 __init__.py**

```python
# src/core/hotword/__init__.py
from src.core.hotword.corrector import PhonemeCorrector, CorrectionResult
from src.core.hotword.phoneme import Phoneme, get_phoneme_info

__all__ = ['PhonemeCorrector', 'CorrectionResult', 'Phoneme', 'get_phoneme_info']
```

**Step 7: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_hotword.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/core/hotword/ tests/test_hotword.py
git commit -m "$(cat <<'EOF'
feat: add phoneme-based hotword correction

- Port phoneme extraction from CapsWriter-Offline
- Implement FastRAG with Numba acceleration
- Add PhonemeCorrector for fuzzy matching
- Support similar phoneme pairs (an/ang, z/zh, etc.)
- Add comprehensive unit tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: 核心服务实现

### Task 4: 实现说话人识别模块

**Files:**
- Create: `src/core/speaker/__init__.py`
- Create: `src/core/speaker/diarization.py`
- Test: `tests/test_speaker.py`

**Step 1: 编写说话人识别测试**

```python
# tests/test_speaker.py
import pytest
from src.core.speaker.diarization import SpeakerLabeler

def test_speaker_labeling():
    """测试说话人标注"""
    labeler = SpeakerLabeler()

    # 模拟带说话人信息的句子
    sentences = [
        {"text": "你好", "start": 0, "end": 1000, "spk": 0},
        {"text": "你好呀", "start": 1200, "end": 2000, "spk": 1},
        {"text": "今天天气不错", "start": 2500, "end": 4000, "spk": 0},
    ]

    result = labeler.label_speakers(sentences)

    assert len(result) == 3
    assert result[0]["speaker"] == "说话人甲"
    assert result[1]["speaker"] == "说话人乙"
    assert result[2]["speaker"] == "说话人甲"

def test_speaker_labels_cycle():
    """测试多说话人标签循环"""
    labeler = SpeakerLabeler()

    # 测试超过两人的情况
    sentences = [
        {"text": "A", "spk": 0},
        {"text": "B", "spk": 1},
        {"text": "C", "spk": 2},
        {"text": "D", "spk": 3},
    ]

    result = labeler.label_speakers(sentences)

    assert result[0]["speaker"] == "说话人甲"
    assert result[1]["speaker"] == "说话人乙"
    assert result[2]["speaker"] == "说话人丙"
    assert result[3]["speaker"] == "说话人丁"
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_speaker.py -v`
Expected: FAIL

**Step 3: 实现说话人标注模块**

```python
# src/core/speaker/diarization.py
"""说话人标注模块"""
from typing import List, Dict, Any, Optional

# 中文说话人标签
SPEAKER_LABELS_ZH = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]


class SpeakerLabeler:
    """说话人标注器"""

    def __init__(self, label_prefix: str = "说话人"):
        self.label_prefix = label_prefix
        self.labels = SPEAKER_LABELS_ZH

    def _get_speaker_label(self, spk_id: int) -> str:
        """获取说话人标签"""
        if spk_id < len(self.labels):
            return f"{self.label_prefix}{self.labels[spk_id]}"
        return f"{self.label_prefix}{spk_id + 1}"

    def label_speakers(
        self,
        sentences: List[Dict[str, Any]],
        spk_key: str = "spk"
    ) -> List[Dict[str, Any]]:
        """为句子添加说话人标签

        Args:
            sentences: 包含 spk 字段的句子列表
            spk_key: 说话人 ID 字段名

        Returns:
            添加了 speaker 字段的句子列表
        """
        result = []
        spk_mapping = {}  # 原始 ID -> 顺序 ID

        for sent in sentences:
            sent_copy = dict(sent)
            spk_id = sent.get(spk_key)

            if spk_id is not None:
                if spk_id not in spk_mapping:
                    spk_mapping[spk_id] = len(spk_mapping)

                mapped_id = spk_mapping[spk_id]
                sent_copy["speaker"] = self._get_speaker_label(mapped_id)
                sent_copy["speaker_id"] = mapped_id
            else:
                sent_copy["speaker"] = "未知"
                sent_copy["speaker_id"] = -1

            result.append(sent_copy)

        return result

    def format_transcript(
        self,
        sentences: List[Dict[str, Any]],
        include_timestamp: bool = True
    ) -> str:
        """格式化为转写文本

        Args:
            sentences: 带标签的句子列表
            include_timestamp: 是否包含时间戳

        Returns:
            格式化的转写文本
        """
        lines = []
        for sent in sentences:
            speaker = sent.get("speaker", "未知")
            text = sent.get("text", "")

            if include_timestamp:
                start = sent.get("start", 0)
                end = sent.get("end", 0)
                timestamp = f"[{self._format_time(start)} - {self._format_time(end)}]"
                lines.append(f"{timestamp} {speaker}: {text}")
            else:
                lines.append(f"{speaker}: {text}")

        return "\n".join(lines)

    @staticmethod
    def _format_time(ms: int) -> str:
        """格式化毫秒为时间字符串"""
        seconds = ms // 1000
        minutes = seconds // 60
        hours = minutes // 60

        if hours > 0:
            return f"{hours:02d}:{minutes % 60:02d}:{seconds % 60:02d}"
        return f"{minutes:02d}:{seconds % 60:02d}"
```

**Step 4: 创建模块 __init__.py**

```python
# src/core/speaker/__init__.py
from src.core.speaker.diarization import SpeakerLabeler, SPEAKER_LABELS_ZH

__all__ = ['SpeakerLabeler', 'SPEAKER_LABELS_ZH']
```

**Step 5: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_speaker.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/core/speaker/ tests/test_speaker.py
git commit -m "$(cat <<'EOF'
feat: add speaker diarization labeling

- Implement SpeakerLabeler for speaker identification
- Support Chinese labels (甲乙丙丁)
- Add transcript formatting with timestamps
- Add unit tests for speaker labeling

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: 实现转写引擎

**Files:**
- Create: `src/core/engine.py`
- Test: `tests/test_engine.py`

**Step 1: 编写转写引擎测试**

```python
# tests/test_engine.py
import pytest
from unittest.mock import Mock, patch
from src.core.engine import TranscriptionEngine

@pytest.fixture
def engine():
    with patch('src.core.engine.model_manager') as mock_mm:
        mock_loader = Mock()
        mock_mm.loader = mock_loader

        # 模拟 ASR 返回结果
        mock_loader.transcribe.return_value = {
            "text": "你好世界",
            "sentence_info": [
                {"text": "你好", "start": 0, "end": 500, "spk": 0},
                {"text": "世界", "start": 600, "end": 1000, "spk": 1},
            ]
        }

        e = TranscriptionEngine()
        yield e

def test_transcribe_basic(engine):
    """测试基本转写"""
    result = engine.transcribe(b"fake_audio_bytes")

    assert result is not None
    assert "text" in result
    assert "sentences" in result

def test_transcribe_with_speaker(engine):
    """测试带说话人识别的转写"""
    result = engine.transcribe(b"fake_audio_bytes", with_speaker=True)

    assert result is not None
    assert len(result["sentences"]) == 2
    assert "speaker" in result["sentences"][0]
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_engine.py -v`
Expected: FAIL

**Step 3: 实现转写引擎**

```python
# src/core/engine.py
"""核心转写引擎"""
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from src.config import settings
from src.models.model_manager import model_manager
from src.core.hotword import PhonemeCorrector
from src.core.speaker import SpeakerLabeler

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """转写引擎 - 整合 ASR + 热词纠错 + 说话人识别"""

    def __init__(self):
        self.corrector = PhonemeCorrector(
            threshold=settings.hotwords_threshold,
            similar_threshold=settings.hotwords_threshold - 0.2
        )
        self.speaker_labeler = SpeakerLabeler()
        self._hotwords_loaded = False

    def load_hotwords(self, path: Optional[str] = None):
        """加载热词"""
        if path is None:
            path = str(settings.hotwords_dir / settings.hotwords_file)

        if Path(path).exists():
            count = self.corrector.load_hotwords_file(path)
            logger.info(f"Loaded {count} hotwords from {path}")
            self._hotwords_loaded = True
        else:
            logger.warning(f"Hotwords file not found: {path}")

    def update_hotwords(self, hotwords: Union[str, List[str]]):
        """更新热词

        Args:
            hotwords: 热词文本 (换行分隔) 或热词列表
        """
        if isinstance(hotwords, list):
            hotwords = "\n".join(hotwords)

        count = self.corrector.update_hotwords(hotwords)
        logger.info(f"Updated {count} hotwords")
        self._hotwords_loaded = True

    def transcribe(
        self,
        audio_input: Union[bytes, str, Path],
        with_speaker: bool = False,
        apply_hotword: bool = True,
        hotwords: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行转写

        Args:
            audio_input: 音频数据 (bytes/路径/URL)
            with_speaker: 是否进行说话人识别
            apply_hotword: 是否应用热词纠错
            hotwords: 本次请求的额外热词 (空格分隔)
            **kwargs: 传递给 FunASR 的额外参数

        Returns:
            转写结果字典
        """
        # 执行 ASR
        try:
            raw_result = model_manager.loader.transcribe(
                audio_input,
                hotwords=hotwords,
                with_speaker=with_speaker,
                **kwargs
            )
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            raise

        # 提取文本和句子信息
        text = raw_result.get("text", "")
        sentence_info = raw_result.get("sentence_info", [])

        # 热词纠错
        if apply_hotword and self._hotwords_loaded and text:
            correction = self.corrector.correct(text)
            text = correction.text

            # 同时纠错每个句子的文本
            for sent in sentence_info:
                sent_correction = self.corrector.correct(sent.get("text", ""))
                sent["text"] = sent_correction.text

        # 说话人标注
        if with_speaker and sentence_info:
            sentence_info = self.speaker_labeler.label_speakers(sentence_info)

        # 构建返回结果
        result = {
            "text": text,
            "sentences": [
                {
                    "text": s.get("text", ""),
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    **({"speaker": s.get("speaker"), "speaker_id": s.get("speaker_id")}
                       if with_speaker else {})
                }
                for s in sentence_info
            ],
            "raw_text": raw_result.get("text", ""),  # 原始文本 (未纠错)
        }

        # 生成格式化转写稿
        if with_speaker:
            result["transcript"] = self.speaker_labeler.format_transcript(
                result["sentences"],
                include_timestamp=True
            )

        return result

    def transcribe_streaming(
        self,
        audio_chunk: bytes,
        cache: Dict[str, Any],
        is_final: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """流式转写 (单个音频块)

        Args:
            audio_chunk: 音频数据块
            cache: 流式状态缓存 (由调用者维护)
            is_final: 是否为最后一块
            **kwargs: 额外参数

        Returns:
            当前块的转写结果
        """
        online_model = model_manager.loader.asr_model_online

        result = online_model.generate(
            input=audio_chunk,
            cache=cache.get("asr_cache", {}),
            is_final=is_final,
            **kwargs
        )

        if result:
            cache["asr_cache"] = result[0].get("cache", {})
            text = result[0].get("text", "")

            # 应用热词纠错
            if self._hotwords_loaded and text:
                correction = self.corrector.correct(text)
                text = correction.text

            return {"text": text, "is_final": is_final}

        return {"text": "", "is_final": is_final}


# 全局引擎实例
transcription_engine = TranscriptionEngine()
```

**Step 4: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/core/engine.py tests/test_engine.py
git commit -m "$(cat <<'EOF'
feat: implement transcription engine

- Integrate ASR + hotword correction + speaker labeling
- Support both batch and streaming transcription
- Add hotwords loading and dynamic update
- Generate formatted transcripts with timestamps

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: API 接口实现

### Task 6: 实现 HTTP API (文件上传转写)

**Files:**
- Create: `src/api/routes/__init__.py`
- Create: `src/api/routes/transcribe.py`
- Create: `src/api/schemas.py`
- Create: `src/api/dependencies.py`
- Test: `tests/test_api_http.py`

**Step 1: 编写 HTTP API 测试**

```python
# tests/test_api_http.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import io

@pytest.fixture
def client():
    with patch('src.core.engine.model_manager'):
        from src.main import app
        with TestClient(app) as c:
            yield c

def test_health_check(client):
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_transcribe_endpoint(client):
    """测试转写接口"""
    with patch('src.core.engine.transcription_engine') as mock_engine:
        mock_engine.transcribe.return_value = {
            "text": "你好世界",
            "sentences": [{"text": "你好世界", "start": 0, "end": 1000}]
        }

        # 创建假音频文件
        audio_content = b"fake_audio_content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}

        response = client.post("/api/v1/transcribe", files=files)

        assert response.status_code == 200
        assert "text" in response.json()
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_api_http.py -v`
Expected: FAIL

**Step 3: 创建 API 模式定义**

```python
# src/api/schemas.py
"""API 请求/响应模式"""
from typing import List, Optional
from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    """转写请求参数"""
    with_speaker: bool = Field(default=False, description="是否进行说话人识别")
    apply_hotword: bool = Field(default=True, description="是否应用热词纠错")
    hotwords: Optional[str] = Field(default=None, description="额外热词 (空格分隔)")


class SentenceInfo(BaseModel):
    """句子信息"""
    text: str = Field(..., description="句子文本")
    start: int = Field(..., description="开始时间 (毫秒)")
    end: int = Field(..., description="结束时间 (毫秒)")
    speaker: Optional[str] = Field(default=None, description="说话人标签")
    speaker_id: Optional[int] = Field(default=None, description="说话人 ID")


class TranscribeResponse(BaseModel):
    """转写响应"""
    code: int = Field(default=0, description="状态码 (0=成功)")
    text: str = Field(..., description="完整转写文本")
    sentences: List[SentenceInfo] = Field(default=[], description="分句信息")
    transcript: Optional[str] = Field(default=None, description="格式化转写稿")
    raw_text: Optional[str] = Field(default=None, description="原始文本 (未纠错)")


class HotwordsUpdateRequest(BaseModel):
    """热词更新请求"""
    hotwords: List[str] = Field(..., description="热词列表")


class HotwordsResponse(BaseModel):
    """热词响应"""
    code: int = Field(default=0)
    count: int = Field(..., description="热词数量")
    message: str = Field(default="success")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
```

**Step 4: 创建依赖注入**

```python
# src/api/dependencies.py
"""API 依赖注入"""
import tempfile
import os
from pathlib import Path
from typing import AsyncGenerator
import aiofiles
from fastapi import UploadFile, HTTPException
import ffmpeg

from src.config import settings


async def process_audio_file(file: UploadFile) -> AsyncGenerator[bytes, None]:
    """处理上传的音频文件，转换为 16kHz PCM"""
    # 保存上传文件到临时目录
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    temp_path = settings.uploads_dir / f"temp_{os.urandom(8).hex()}{suffix}"

    try:
        # 写入临时文件
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # 使用 ffmpeg 转换为标准格式
        try:
            audio_bytes, _ = (
                ffmpeg
                .input(str(temp_path), threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
            yield audio_bytes
        except ffmpeg.Error as e:
            raise HTTPException(
                status_code=400,
                detail=f"音频处理失败: {e.stderr.decode() if e.stderr else str(e)}"
            )
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
```

**Step 5: 实现转写路由**

```python
# src/api/routes/transcribe.py
"""转写 API 路由"""
import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Optional

from src.api.schemas import TranscribeResponse, SentenceInfo
from src.api.dependencies import process_audio_file
from src.core.engine import transcription_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["transcribe"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="音频文件"),
    with_speaker: bool = Form(default=False, description="是否进行说话人识别"),
    apply_hotword: bool = Form(default=True, description="是否应用热词纠错"),
    hotwords: Optional[str] = Form(default=None, description="额外热词 (空格分隔)"),
):
    """
    上传音频文件进行转写

    支持的音频格式: wav, mp3, m4a, flac, ogg 等 (ffmpeg 支持的格式)

    - **with_speaker**: 启用说话人识别，结果中会包含说话人标签
    - **apply_hotword**: 启用热词纠错，提高专有名词识别准确率
    - **hotwords**: 本次请求的临时热词，空格分隔
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="请上传音频文件")

    try:
        # 处理音频文件
        async for audio_bytes in process_audio_file(file):
            # 执行转写
            result = transcription_engine.transcribe(
                audio_bytes,
                with_speaker=with_speaker,
                apply_hotword=apply_hotword,
                hotwords=hotwords,
            )

            return TranscribeResponse(
                code=0,
                text=result["text"],
                sentences=[SentenceInfo(**s) for s in result["sentences"]],
                transcript=result.get("transcript"),
                raw_text=result.get("raw_text"),
            )

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"转写失败: {str(e)}")
```

**Step 6: 创建路由 __init__.py**

```python
# src/api/routes/__init__.py
from fastapi import APIRouter
from src.api.routes.transcribe import router as transcribe_router

api_router = APIRouter()
api_router.include_router(transcribe_router)

__all__ = ['api_router']
```

**Step 7: 更新主应用**

```python
# src/main.py
"""TingWu Speech Service 主入口"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.api.routes import api_router
from src.api.schemas import HealthResponse
from src.core.engine import transcription_engine
from src.models.model_manager import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    # 预加载模型
    logger.info("Preloading models...")
    model_manager.preload_models(with_speaker=True)

    # 加载热词
    transcription_engine.load_hotwords()

    logger.info("Service ready!")

    yield

    # 关闭时
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="基于 FunASR + CapsWriter 的中文语音转写服务",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """健康检查"""
    return HealthResponse(status="healthy", version=settings.version)


@app.get("/", tags=["system"])
async def root():
    """服务信息"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
    }
```

**Step 8: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_api_http.py -v`
Expected: PASS

**Step 9: Commit**

```bash
git add src/api/ src/main.py tests/test_api_http.py
git commit -m "$(cat <<'EOF'
feat: implement HTTP API for file transcription

- Add /api/v1/transcribe endpoint for audio upload
- Support speaker diarization and hotword correction
- Add Pydantic schemas for request/response validation
- Add health check endpoint
- Configure CORS middleware

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: 实现 WebSocket API (实时流式转写)

**Files:**
- Create: `src/api/routes/websocket.py`
- Create: `src/api/ws_manager.py`
- Test: `tests/test_api_websocket.py`

**Step 1: 编写 WebSocket 测试**

```python
# tests/test_api_websocket.py
import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock

def test_ws_connection_state():
    """测试 WebSocket 连接状态管理"""
    from src.api.ws_manager import ConnectionState

    state = ConnectionState()
    assert state.is_speaking == False
    assert state.asr_cache == {}

    state.is_speaking = True
    assert state.is_speaking == True

def test_ws_manager_add_remove():
    """测试连接管理器添加/移除"""
    from src.api.ws_manager import WebSocketManager

    manager = WebSocketManager()
    mock_ws = Mock()

    # 添加连接
    manager.connect(mock_ws, "test-id")
    assert "test-id" in manager.connections

    # 移除连接
    manager.disconnect("test-id")
    assert "test-id" not in manager.connections
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_api_websocket.py -v`
Expected: FAIL

**Step 3: 实现 WebSocket 管理器**

```python
# src/api/ws_manager.py
"""WebSocket 连接管理"""
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class ConnectionState:
    """WebSocket 连接状态"""
    is_speaking: bool = False
    asr_cache: Dict[str, Any] = field(default_factory=dict)
    vad_cache: Dict[str, Any] = field(default_factory=dict)
    chunk_interval: int = 10
    mode: str = "2pass"  # online | offline | 2pass
    hotwords: Optional[str] = None

    def reset(self):
        """重置状态"""
        self.is_speaking = False
        self.asr_cache = {}
        self.vad_cache = {}


class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.states: Dict[str, ConnectionState] = {}

    def connect(self, websocket: WebSocket, connection_id: str):
        """添加新连接"""
        self.connections[connection_id] = websocket
        self.states[connection_id] = ConnectionState()
        logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str):
        """移除连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        if connection_id in self.states:
            del self.states[connection_id]
        logger.info(f"WebSocket disconnected: {connection_id}")

    def get_state(self, connection_id: str) -> Optional[ConnectionState]:
        """获取连接状态"""
        return self.states.get(connection_id)

    async def send_json(self, connection_id: str, data: Dict[str, Any]):
        """发送 JSON 消息"""
        websocket = self.connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")

    async def broadcast(self, data: Dict[str, Any]):
        """广播消息给所有连接"""
        for connection_id in list(self.connections.keys()):
            await self.send_json(connection_id, data)


# 全局管理器实例
ws_manager = WebSocketManager()
```

**Step 4: 实现 WebSocket 路由**

```python
# src/api/routes/websocket.py
"""WebSocket 实时转写路由"""
import json
import uuid
import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.ws_manager import ws_manager, ConnectionState
from src.core.engine import transcription_engine
from src.models.model_manager import model_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    实时流式转写 WebSocket 接口

    协议说明:

    1. 客户端发送配置消息 (JSON):
    ```json
    {
        "is_speaking": true,
        "mode": "2pass",
        "hotwords": "可选热词",
        "chunk_interval": 10
    }
    ```

    2. 客户端发送音频数据 (binary):
    - 格式: PCM 16bit, 16kHz, mono
    - 建议每 600ms 发送一次 (9600 bytes)

    3. 服务端返回识别结果 (JSON):
    ```json
    {
        "mode": "2pass-online",
        "text": "识别文本",
        "is_final": false
    }
    ```
    """
    await websocket.accept()

    connection_id = str(uuid.uuid4())
    ws_manager.connect(websocket, connection_id)
    state = ws_manager.get_state(connection_id)

    # 初始化模型
    model_online = model_manager.loader.asr_model_online
    model_offline = model_manager.loader.asr_model

    frames = []
    frames_asr = []
    frames_online = []

    try:
        while True:
            message = await websocket.receive()

            # 处理文本消息 (配置)
            if "text" in message:
                try:
                    config = json.loads(message["text"])
                    _handle_config(state, config)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON config: {message['text']}")
                continue

            # 处理二进制消息 (音频)
            if "bytes" in message:
                audio_chunk = message["bytes"]
                frames.append(audio_chunk)
                frames_online.append(audio_chunk)

                # 在线识别
                if len(frames_online) % state.chunk_interval == 0:
                    if state.mode in ("2pass", "online"):
                        audio_in = b"".join(frames_online)
                        result = await _asr_online(
                            model_online, audio_in, state
                        )
                        if result and result.get("text"):
                            await websocket.send_json({
                                "mode": "2pass-online" if state.mode == "2pass" else "online",
                                "text": result["text"],
                                "is_final": False,
                            })
                        frames_online = []

                # VAD 结束时执行离线识别
                if not state.is_speaking:
                    if state.mode in ("2pass", "offline"):
                        audio_in = b"".join(frames)
                        result = await _asr_offline(
                            model_offline, audio_in, state
                        )
                        if result and result.get("text"):
                            # 应用热词纠错
                            text = result["text"]
                            if transcription_engine._hotwords_loaded:
                                correction = transcription_engine.corrector.correct(text)
                                text = correction.text

                            await websocket.send_json({
                                "mode": "2pass-offline" if state.mode == "2pass" else "offline",
                                "text": text,
                                "is_final": True,
                            })

                    # 重置缓存
                    frames = []
                    frames_online = []
                    state.asr_cache = {}

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        ws_manager.disconnect(connection_id)


def _handle_config(state: ConnectionState, config: dict):
    """处理配置消息"""
    if "is_speaking" in config:
        state.is_speaking = config["is_speaking"]
    if "mode" in config:
        state.mode = config["mode"]
    if "hotwords" in config:
        state.hotwords = config["hotwords"]
    if "chunk_interval" in config:
        state.chunk_interval = config["chunk_interval"]


async def _asr_online(model, audio_in: bytes, state: ConnectionState) -> dict:
    """在线流式识别"""
    try:
        result = model.generate(
            input=audio_in,
            cache=state.asr_cache,
            is_final=not state.is_speaking,
            hotword=state.hotwords,
        )
        if result:
            state.asr_cache = result[0].get("cache", {})
            return {"text": result[0].get("text", "")}
    except Exception as e:
        logger.error(f"Online ASR error: {e}")
    return {}


async def _asr_offline(model, audio_in: bytes, state: ConnectionState) -> dict:
    """离线识别"""
    try:
        result = model.generate(
            input=audio_in,
            hotword=state.hotwords,
        )
        if result:
            return {"text": result[0].get("text", "")}
    except Exception as e:
        logger.error(f"Offline ASR error: {e}")
    return {}
```

**Step 5: 更新路由注册**

```python
# src/api/routes/__init__.py (更新)
from fastapi import APIRouter
from src.api.routes.transcribe import router as transcribe_router
from src.api.routes.websocket import router as websocket_router

api_router = APIRouter()
api_router.include_router(transcribe_router)
api_router.include_router(websocket_router)

__all__ = ['api_router']
```

**Step 6: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_api_websocket.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/api/routes/websocket.py src/api/ws_manager.py tests/test_api_websocket.py
git commit -m "$(cat <<'EOF'
feat: implement WebSocket API for real-time transcription

- Add /ws/realtime endpoint for streaming ASR
- Support 2pass mode (online + offline refinement)
- Add ConnectionState for per-connection caching
- Integrate hotword correction in final results

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: 热词与管理 API

### Task 8: 实现热词管理 API

**Files:**
- Create: `src/api/routes/hotwords.py`
- Update: `src/api/routes/__init__.py`
- Test: `tests/test_api_hotwords.py`

**Step 1: 编写热词 API 测试**

```python
# tests/test_api_hotwords.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

@pytest.fixture
def client():
    with patch('src.core.engine.model_manager'):
        from src.main import app
        with TestClient(app) as c:
            yield c

def test_get_hotwords(client):
    """测试获取热词列表"""
    response = client.get("/api/v1/hotwords")
    assert response.status_code == 200
    assert "hotwords" in response.json()

def test_update_hotwords(client):
    """测试更新热词"""
    response = client.post(
        "/api/v1/hotwords",
        json={"hotwords": ["Claude", "Bilibili", "麦当劳"]}
    )
    assert response.status_code == 200
    assert response.json()["count"] == 3
```

**Step 2: 运行测试验证失败**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_api_hotwords.py -v`
Expected: FAIL

**Step 3: 实现热词管理路由**

```python
# src/api/routes/hotwords.py
"""热词管理 API"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.core.engine import transcription_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hotwords", tags=["hotwords"])


class HotwordsListResponse(BaseModel):
    code: int = 0
    hotwords: List[str] = Field(..., description="热词列表")
    count: int = Field(..., description="热词数量")


class HotwordsUpdateRequest(BaseModel):
    hotwords: List[str] = Field(..., description="热词列表")


class HotwordsUpdateResponse(BaseModel):
    code: int = 0
    count: int = Field(..., description="更新后的热词数量")
    message: str = "success"


@router.get("", response_model=HotwordsListResponse)
async def get_hotwords():
    """获取当前热词列表"""
    hotwords = list(transcription_engine.corrector.hotwords.keys())
    return HotwordsListResponse(
        hotwords=hotwords,
        count=len(hotwords)
    )


@router.post("", response_model=HotwordsUpdateResponse)
async def update_hotwords(request: HotwordsUpdateRequest):
    """更新热词列表

    会替换当前所有热词
    """
    try:
        transcription_engine.update_hotwords(request.hotwords)
        return HotwordsUpdateResponse(
            count=len(request.hotwords),
            message="热词更新成功"
        )
    except Exception as e:
        logger.error(f"Failed to update hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/append", response_model=HotwordsUpdateResponse)
async def append_hotwords(request: HotwordsUpdateRequest):
    """追加热词

    保留现有热词，追加新热词
    """
    try:
        existing = list(transcription_engine.corrector.hotwords.keys())
        combined = list(set(existing + request.hotwords))
        transcription_engine.update_hotwords(combined)
        return HotwordsUpdateResponse(
            count=len(combined),
            message=f"追加了 {len(request.hotwords)} 个热词"
        )
    except Exception as e:
        logger.error(f"Failed to append hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload", response_model=HotwordsUpdateResponse)
async def reload_hotwords():
    """从文件重新加载热词"""
    try:
        transcription_engine.load_hotwords()
        count = len(transcription_engine.corrector.hotwords)
        return HotwordsUpdateResponse(
            count=count,
            message="热词重新加载成功"
        )
    except Exception as e:
        logger.error(f"Failed to reload hotwords: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: 更新路由注册**

```python
# src/api/routes/__init__.py (最终版本)
from fastapi import APIRouter
from src.api.routes.transcribe import router as transcribe_router
from src.api.routes.websocket import router as websocket_router
from src.api.routes.hotwords import router as hotwords_router

api_router = APIRouter()
api_router.include_router(transcribe_router)
api_router.include_router(websocket_router)
api_router.include_router(hotwords_router)

__all__ = ['api_router']
```

**Step 5: 运行测试验证通过**

Run: `cd /data/temp41/tingwu && python -m pytest tests/test_api_hotwords.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/api/routes/hotwords.py tests/test_api_hotwords.py
git commit -m "$(cat <<'EOF'
feat: add hotwords management API

- GET /api/v1/hotwords - list current hotwords
- POST /api/v1/hotwords - replace all hotwords
- POST /api/v1/hotwords/append - add new hotwords
- POST /api/v1/hotwords/reload - reload from file

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Docker 部署配置

### Task 9: 完善 Docker 部署配置

**Files:**
- Update: `Dockerfile`
- Update: `docker-compose.yml`
- Create: `docker-compose.cpu.yml`
- Create: `scripts/start.sh`
- Create: `data/hotwords/hotwords.txt`

**Step 1: 创建示例热词文件**

```text
# data/hotwords/hotwords.txt
# 常用热词配置
# 每行一个热词，# 开头为注释

# 技术词汇
FunASR
CapsWriter
Paraformer
Python
JavaScript
TypeScript
Docker
Kubernetes

# 公司/产品
阿里巴巴
腾讯
字节跳动
百度
华为
小米
Bilibili

# 中文常见误识别
麦当劳
肯德基
星巴克
```

**Step 2: 更新 Dockerfile (支持多阶段构建)**

```dockerfile
# Dockerfile (更新版)
# 第一阶段: 构建依赖
FROM python:3.10-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 第二阶段: 运行镜像
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制已安装的依赖
COPY --from=builder /install /usr/local

WORKDIR /app

# 复制源代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/hotwords/ ./data/hotwords/

# 创建数据目录
RUN mkdir -p data/models data/uploads data/outputs

# 非 root 用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: 更新 docker-compose.yml (GPU 版本)**

```yaml
# docker-compose.yml
version: '3.8'

services:
  tingwu:
    build:
      context: .
      dockerfile: Dockerfile
    image: tingwu-speech-service:latest
    container_name: tingwu-service
    ports:
      - "${PORT:-8000}:8000"
    volumes:
      - ./data:/app/data
      - model-cache:/root/.cache/modelscope
      - huggingface-cache:/root/.cache/huggingface
    environment:
      - DEVICE=${DEVICE:-cuda}
      - NGPU=${NGPU:-1}
      - NCPU=${NCPU:-4}
      - DEBUG=${DEBUG:-false}
      - HOTWORDS_THRESHOLD=${HOTWORDS_THRESHOLD:-0.85}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

volumes:
  model-cache:
  huggingface-cache:
```

**Step 4: 创建 CPU 版本 docker-compose**

```yaml
# docker-compose.cpu.yml
version: '3.8'

services:
  tingwu:
    build:
      context: .
      dockerfile: Dockerfile
    image: tingwu-speech-service:cpu
    container_name: tingwu-service-cpu
    ports:
      - "${PORT:-8000}:8000"
    volumes:
      - ./data:/app/data
      - model-cache:/root/.cache/modelscope
      - huggingface-cache:/root/.cache/huggingface
    environment:
      - DEVICE=cpu
      - NGPU=0
      - NCPU=${NCPU:-8}
      - DEBUG=${DEBUG:-false}
      - HOTWORDS_THRESHOLD=${HOTWORDS_THRESHOLD:-0.85}
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

volumes:
  model-cache:
  huggingface-cache:
```

**Step 5: 创建启动脚本**

```bash
#!/bin/bash
# scripts/start.sh

set -e

# 默认配置
MODE=${1:-gpu}
PORT=${PORT:-8000}

echo "======================================"
echo "TingWu Speech Service Launcher"
echo "======================================"

case $MODE in
    gpu)
        echo "Starting with GPU support..."
        docker compose up -d
        ;;
    cpu)
        echo "Starting with CPU only..."
        docker compose -f docker-compose.cpu.yml up -d
        ;;
    build)
        echo "Building Docker image..."
        docker compose build
        ;;
    stop)
        echo "Stopping service..."
        docker compose down
        docker compose -f docker-compose.cpu.yml down 2>/dev/null || true
        ;;
    logs)
        docker compose logs -f
        ;;
    *)
        echo "Usage: $0 {gpu|cpu|build|stop|logs}"
        exit 1
        ;;
esac

echo ""
echo "Service URL: http://localhost:${PORT}"
echo "API Docs: http://localhost:${PORT}/docs"
echo "WebSocket: ws://localhost:${PORT}/ws/realtime"
```

**Step 6: 设置脚本权限**

Run: `chmod +x scripts/start.sh`
Expected: 成功

**Step 7: Commit**

```bash
git add Dockerfile docker-compose.yml docker-compose.cpu.yml scripts/ data/hotwords/
git commit -m "$(cat <<'EOF'
chore: finalize Docker deployment configuration

- Multi-stage Dockerfile for smaller image size
- Separate GPU and CPU docker-compose files
- Add startup script with multiple modes
- Include default hotwords file
- Add health check and logging configuration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: 创建 API 文档和使用示例

**Files:**
- Create: `docs/API.md`
- Create: `examples/client_http.py`
- Create: `examples/client_websocket.py`

**Step 1: 创建 API 文档**

```markdown
# docs/API.md
# TingWu Speech Service API 文档

## 概述

TingWu 是基于 FunASR + CapsWriter-Offline 的中文语音转写服务，支持：
- 文件上传批量转写
- 实时流式转写
- 说话人识别（甲/乙/丙/丁标注）
- 热词纠错

## 基础信息

- **基础 URL**: `http://localhost:8000`
- **API 版本**: v1
- **数据格式**: JSON

---

## HTTP API

### 1. 健康检查

```http
GET /health
```

**响应示例**:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. 音频转写

```http
POST /api/v1/transcribe
Content-Type: multipart/form-data
```

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 音频文件 (支持 wav/mp3/m4a 等) |
| with_speaker | bool | 否 | 启用说话人识别 (默认 false) |
| apply_hotword | bool | 否 | 启用热词纠错 (默认 true) |
| hotwords | string | 否 | 临时热词 (空格分隔) |

**响应示例**:
```json
{
  "code": 0,
  "text": "你好，今天天气不错。是的，很适合出去玩。",
  "sentences": [
    {
      "text": "你好，今天天气不错。",
      "start": 0,
      "end": 2500,
      "speaker": "说话人甲",
      "speaker_id": 0
    },
    {
      "text": "是的，很适合出去玩。",
      "start": 3000,
      "end": 5000,
      "speaker": "说话人乙",
      "speaker_id": 1
    }
  ],
  "transcript": "[00:00 - 00:02] 说话人甲: 你好，今天天气不错。\n[00:03 - 00:05] 说话人乙: 是的，很适合出去玩。"
}
```

### 3. 热词管理

#### 获取热词列表

```http
GET /api/v1/hotwords
```

#### 更新热词

```http
POST /api/v1/hotwords
Content-Type: application/json

{
  "hotwords": ["Claude", "FunASR", "麦当劳"]
}
```

#### 追加热词

```http
POST /api/v1/hotwords/append
Content-Type: application/json

{
  "hotwords": ["新热词1", "新热词2"]
}
```

---

## WebSocket API

### 实时流式转写

```
WebSocket: ws://localhost:8000/ws/realtime
```

#### 消息协议

**1. 客户端发送配置 (JSON)**:
```json
{
  "is_speaking": true,
  "mode": "2pass",
  "hotwords": "可选热词",
  "chunk_interval": 10
}
```

**2. 客户端发送音频 (Binary)**:
- 格式: PCM 16bit, 单声道, 16kHz
- 建议每 600ms 发送一次 (约 19200 bytes)

**3. 服务端返回结果 (JSON)**:
```json
{
  "mode": "2pass-online",
  "text": "你好",
  "is_final": false
}
```

#### 识别模式

| 模式 | 说明 |
|------|------|
| online | 仅实时识别，延迟低 |
| offline | 仅离线识别，精度高 |
| 2pass | 先实时后离线 (推荐) |

---

## 错误码

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |
```

**Step 2: 创建 HTTP 客户端示例**

```python
# examples/client_http.py
"""HTTP 客户端示例"""
import requests
import sys

API_BASE = "http://localhost:8000"


def transcribe_file(file_path: str, with_speaker: bool = False):
    """上传音频文件进行转写"""
    url = f"{API_BASE}/api/v1/transcribe"

    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        data = {
            "with_speaker": with_speaker,
            "apply_hotword": True,
        }

        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"转写结果: {result['text']}")
        print()

        if with_speaker and result.get("transcript"):
            print("带说话人的转写稿:")
            print(result["transcript"])

        return result
    else:
        print(f"转写失败: {response.text}")
        return None


def update_hotwords(hotwords: list):
    """更新热词"""
    url = f"{API_BASE}/api/v1/hotwords"
    response = requests.post(url, json={"hotwords": hotwords})

    if response.status_code == 200:
        print(f"热词更新成功: {response.json()}")
    else:
        print(f"热词更新失败: {response.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python client_http.py <音频文件路径> [--speaker]")
        sys.exit(1)

    file_path = sys.argv[1]
    with_speaker = "--speaker" in sys.argv

    transcribe_file(file_path, with_speaker)
```

**Step 3: 创建 WebSocket 客户端示例**

```python
# examples/client_websocket.py
"""WebSocket 实时转写客户端示例"""
import asyncio
import json
import wave
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("请安装 websockets: pip install websockets")
    sys.exit(1)

WS_URL = "ws://localhost:8000/ws/realtime"
CHUNK_SIZE = 9600  # 600ms @ 16kHz


async def realtime_transcribe(audio_path: str):
    """实时转写音频文件"""

    # 读取音频文件
    with wave.open(audio_path, "rb") as wf:
        assert wf.getnchannels() == 1, "需要单声道音频"
        assert wf.getsampwidth() == 2, "需要 16bit 音频"
        assert wf.getframerate() == 16000, "需要 16kHz 采样率"

        audio_data = wf.readframes(wf.getnframes())

    print(f"音频长度: {len(audio_data) / 32000:.1f} 秒")
    print("开始实时转写...")
    print("-" * 40)

    async with websockets.connect(WS_URL, subprotocols=["binary"]) as ws:
        # 发送配置
        config = {
            "is_speaking": True,
            "mode": "2pass",
            "chunk_interval": 10,
        }
        await ws.send(json.dumps(config))

        # 分块发送音频
        for i in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[i:i + CHUNK_SIZE]
            await ws.send(chunk)

            # 接收结果
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                result = json.loads(response)
                mode = result.get("mode", "")
                text = result.get("text", "")
                is_final = result.get("is_final", False)

                if text:
                    prefix = "[最终]" if is_final else "[实时]"
                    print(f"{prefix} {text}")
            except asyncio.TimeoutError:
                pass

            # 模拟实时发送间隔
            await asyncio.sleep(0.3)

        # 标记说话结束
        await ws.send(json.dumps({"is_speaking": False}))

        # 等待最终结果
        try:
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            result = json.loads(response)
            if result.get("text"):
                print(f"[最终] {result['text']}")
        except asyncio.TimeoutError:
            pass

    print("-" * 40)
    print("转写完成")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python client_websocket.py <音频文件路径>")
        print("注意: 音频需要是 16kHz, 单声道, 16bit PCM WAV 格式")
        sys.exit(1)

    audio_path = sys.argv[1]
    asyncio.run(realtime_transcribe(audio_path))
```

**Step 4: Commit**

```bash
git add docs/API.md examples/
git commit -m "$(cat <<'EOF'
docs: add API documentation and client examples

- Add comprehensive API documentation (docs/API.md)
- Add HTTP client example (examples/client_http.py)
- Add WebSocket client example (examples/client_websocket.py)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: 测试与完善

### Task 11: 集成测试

**Files:**
- Create: `tests/test_integration.py`
- Create: `conftest.py`

**Step 1: 创建测试配置**

```python
# conftest.py
import pytest
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

@pytest.fixture(scope="session")
def test_audio_path():
    """测试音频路径"""
    return Path(__file__).parent / "tests" / "fixtures" / "test.wav"
```

**Step 2: 创建集成测试**

```python
# tests/test_integration.py
"""集成测试"""
import pytest
from unittest.mock import patch, Mock

def test_full_pipeline():
    """测试完整转写流程"""
    with patch('src.models.model_manager.model_manager') as mock_mm:
        mock_loader = Mock()
        mock_mm.loader = mock_loader
        mock_loader.transcribe.return_value = {
            "text": "买当劳很好吃",
            "sentence_info": [
                {"text": "买当劳很好吃", "start": 0, "end": 1500, "spk": 0}
            ]
        }

        from src.core.engine import TranscriptionEngine

        engine = TranscriptionEngine()
        engine.update_hotwords(["麦当劳", "肯德基"])

        result = engine.transcribe(
            b"fake_audio",
            with_speaker=True,
            apply_hotword=True
        )

        # 验证热词纠错生效
        assert "麦当劳" in result["text"]
        # 验证说话人标注
        assert result["sentences"][0]["speaker"] == "说话人甲"

def test_hotword_correction_accuracy():
    """测试热词纠错准确率"""
    from src.core.hotword import PhonemeCorrector

    corrector = PhonemeCorrector(threshold=0.8)
    corrector.update_hotwords("Claude\nBilibili\n麦当劳\n肯德基\nFunASR")

    test_cases = [
        ("Hello klaude", "Claude"),
        ("我要去买当劳", "麦当劳"),
        ("肯得鸡真好吃", "肯德基"),
        ("喜欢看bili bili", "Bilibili"),
    ]

    for input_text, expected_word in test_cases:
        result = corrector.correct(input_text)
        assert expected_word in result.text, f"Failed: {input_text} -> {result.text}"
```

**Step 3: 运行所有测试**

Run: `cd /data/temp41/tingwu && python -m pytest tests/ -v --tb=short`
Expected: 所有测试通过

**Step 4: Commit**

```bash
git add tests/test_integration.py conftest.py
git commit -m "$(cat <<'EOF'
test: add integration tests

- Test full transcription pipeline
- Test hotword correction accuracy
- Add pytest configuration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: 最终文档和清理

**Files:**
- Create: `README.md`
- Create: `.gitignore`

**Step 1: 创建 README**

```markdown
# README.md
# TingWu 语音转写服务

基于 FunASR + CapsWriter-Offline 的高精度中文语音转写服务。

## 特性

- **高精度转写**: 基于阿里 FunASR Paraformer-large 模型
- **热词纠错**: 音素级模糊匹配，自动纠正专有名词
- **说话人识别**: 自动识别多说话人并标注（甲/乙/丙/丁）
- **实时转写**: WebSocket 流式接口，支持 2pass 模式
- **Docker 部署**: 支持 GPU/CPU 部署

## 快速开始

### Docker 部署 (推荐)

```bash
# GPU 版本
./scripts/start.sh gpu

# CPU 版本
./scripts/start.sh cpu

# 查看日志
./scripts/start.sh logs
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn src.main:app --reload
```

## API 使用

### 文件转写

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav" \
  -F "with_speaker=true"
```

### 实时转写

参见 `examples/client_websocket.py`

## 配置

通过环境变量或 `.env` 文件配置:

| 变量 | 说明 | 默认值 |
|------|------|--------|
| DEVICE | 设备类型 | cuda |
| NGPU | GPU 数量 | 1 |
| HOTWORDS_THRESHOLD | 热词匹配阈值 | 0.85 |

## 文档

- [API 文档](docs/API.md)
- [实现计划](docs/plans/)

## License

MIT
```

**Step 2: 创建 .gitignore**

```text
# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
.eggs/
dist/
build/

# Environment
.env
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Data
data/uploads/*
data/outputs/*
data/models/*
!data/hotwords/hotwords.txt
*.wav
*.mp3
*.m4a

# Docker
*.log

# Cache
.cache/
.pytest_cache/
.mypy_cache/

# OS
.DS_Store
Thumbs.db
```

**Step 3: 最终 Commit**

```bash
git add README.md .gitignore
git commit -m "$(cat <<'EOF'
docs: add README and gitignore

- Complete project README with quick start guide
- Add comprehensive .gitignore

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## 完成清单

| Phase | Task | 描述 |
|-------|------|------|
| 1 | Task 1 | 项目目录结构 |
| 1 | Task 2 | FunASR 模型加载器 |
| 1 | Task 3 | 热词纠错模块 |
| 2 | Task 4 | 说话人识别模块 |
| 2 | Task 5 | 转写引擎 |
| 3 | Task 6 | HTTP API |
| 3 | Task 7 | WebSocket API |
| 4 | Task 8 | 热词管理 API |
| 5 | Task 9 | Docker 部署 |
| 5 | Task 10 | API 文档 |
| 6 | Task 11 | 集成测试 |
| 6 | Task 12 | 最终文档 |

---

## 参考资源

- [FunASR 官方文档](https://github.com/modelscope/FunASR)
- [CAM++ 说话人日志](https://modelscope.cn/models/damo/speech_campplus_speaker-diarization_common)
- [CapsWriter-Offline 热词系统](https://github.com/HaujetZhao/CapsWriter-Offline)
- [FunASR 实时多说话人分离讨论](https://github.com/modelscope/FunASR/issues/2715)
