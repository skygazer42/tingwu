# TingWu 语音转写服务

基于 FunASR + CapsWriter-Offline 的高精度中文语音转写服务。

## 特性

- **高精度转写**: 基于阿里 FunASR Paraformer-large 模型 (60000小时训练数据)
- **热词纠错**: 音素级模糊匹配，自动纠正专有名词 (基于 CapsWriter-Offline)
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

# 停止服务
./scripts/start.sh stop
```

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 使用

### 文件转写

```bash
# 基本转写
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav"

# 带说话人识别
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio.wav" \
  -F "with_speaker=true"
```

### 实时转写

参见 `examples/client_websocket.py`

### 热词管理

```bash
# 查看热词
curl http://localhost:8000/api/v1/hotwords

# 更新热词
curl -X POST http://localhost:8000/api/v1/hotwords \
  -H "Content-Type: application/json" \
  -d '{"hotwords": ["Claude", "FunASR", "麦当劳"]}'
```

### API 文档

启动服务后访问: http://localhost:8000/docs

## 配置

通过环境变量或 `.env` 文件配置:

| 变量 | 说明 | 默认值 |
|------|------|--------|
| DEVICE | 设备类型 (cuda/cpu) | cuda |
| NGPU | GPU 数量 | 1 |
| NCPU | CPU 线程数 | 4 |
| PORT | 服务端口 | 8000 |
| HOTWORDS_THRESHOLD | 热词匹配阈值 | 0.85 |

## 项目结构

```
tingwu/
├── src/
│   ├── api/            # FastAPI 接口层
│   │   ├── routes/     # 路由定义
│   │   ├── schemas.py  # 数据模型
│   │   └── ws_manager.py # WebSocket 管理
│   ├── core/           # 核心业务逻辑
│   │   ├── hotword/    # 热词纠错 (CapsWriter)
│   │   ├── speaker/    # 说话人识别
│   │   └── engine.py   # 转写引擎
│   ├── models/         # 模型加载
│   ├── config.py       # 配置管理
│   └── main.py         # 应用入口
├── data/hotwords/      # 热词配置
├── examples/           # 使用示例
├── tests/              # 测试
├── Dockerfile
├── docker-compose.yml  # GPU 部署
└── docker-compose.cpu.yml # CPU 部署
```

## 技术栈

- **ASR**: FunASR (Paraformer-large + FSMN-VAD + CT-Transformer)
- **说话人**: CAMPPlus + ClusterBackend
- **热词**: CapsWriter-Offline 音素 RAG
- **服务**: FastAPI + WebSocket
- **部署**: Docker + Docker Compose
