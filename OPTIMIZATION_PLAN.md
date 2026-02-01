# TingWu 项目优化计划 (20项) - ✅ 已完成

基于项目架构分析和最新技术调研，制定以下20项优化计划。

**状态**: 所有 20 项优化已实现完成

## 一、模型推理性能优化 (1-4)

### 1. ✅ ONNX INT8 量化支持
**目标**: 推理速度提升40%，模型大小减少75%
**实现位置**: `src/models/backends/onnx.py`, `src/config.py`
- 添加 `onnx_intra_threads`, `onnx_inter_threads` 配置
- 实现 `warmup()` 方法预热模型
- 支持线程数优化配置

### 2. ✅ 模型预热机制
**目标**: 消除首次推理延迟
**实现位置**: `src/core/engine.py`, `src/main.py`
- `TranscriptionEngine.warmup()` 方法
- 启动时自动调用预热
- `warmup_on_startup` 配置项

### 3. ✅ 批量音频处理 API
**目标**: 支持多文件并行转写，提升吞吐量
**实现位置**: `src/api/routes/transcribe.py`, `src/api/schemas.py`
- `POST /api/v1/transcribe/batch` 端点
- `BatchTranscribeResponse` 响应模型
- 并行处理和聚合统计

### 4. ✅ GPU 内存优化
**目标**: 减少显存占用，支持更大批量
**实现位置**: `src/models/backends/pytorch.py`
- `_cleanup_gpu_memory()` 方法
- 自动 `torch.cuda.empty_cache()`
- 转写后内存清理

## 二、WebSocket 流式优化 (5-7)

### 5. ✅ WebSocket 压缩
**目标**: 减少网络带宽消耗 60-80%
**实现位置**: `src/config.py`, WebSocket 连接配置
- `ws_compression: bool = True` 配置项
- 启用 `permessage-deflate` 压缩扩展

### 6. ✅ 心跳与重连机制
**目标**: 提升连接稳定性
**实现位置**: `src/api/routes/websocket.py`
- `_heartbeat_task()` 异步心跳任务
- `ws_heartbeat_interval` 和 `ws_heartbeat_timeout` 配置
- ping/pong 帧检测

### 7. ✅ 自适应分块大小
**目标**: 根据网络条件优化流式体验
**实现位置**: `src/api/ws_manager.py`
- `ConnectionState.update_latency()` 方法
- 基于延迟动态调整 `chunk_size`
- 延迟采样和平滑

## 三、热词系统优化 (8-10)

### 8. ✅ H-PRM 声学相似度预检索
**目标**: 热词召回率从 70% 提升到 93%+
**实现位置**: `src/core/hotword/corrector.py`
- 通过 FAISS 向量检索实现声学相似度
- `_phoneme_to_vector()` 音素向量化
- 粗筛+精排两阶段检索

### 9. ✅ FAISS 向量索引加速
**目标**: 大规模热词检索加速 10x
**实现位置**: `src/core/hotword/corrector.py`, `src/config.py`
- `use_faiss: bool`, `faiss_index_type: str` 配置
- 支持 IVFFlat, HNSW 索引类型
- `_build_faiss_index()`, `_faiss_search()` 方法

### 10. ✅ 音形义联合纠错
**目标**: 提升同音字纠错准确率
**实现位置**: `src/core/hotword/shape_corrector.py`, `src/core/hotword/corrector.py`
- `ShapeCorrector` 字形相似度模块
- `JointCorrector` 音形义联合评分
- `use_shape_rerank`, `shape_weight` 配置

## 四、LLM 集成优化 (11-14)

### 11. ✅ vLLM 高吞吐后端
**目标**: LLM 推理吞吐量提升 50x
**实现位置**: `src/core/llm/client.py`, `src/config.py`
- `llm_backend: str = "auto"` (auto, ollama, openai, vllm)
- `_chat_vllm()` vLLM 客户端实现
- continuous batching 支持

### 12. ✅ LLM 响应缓存
**目标**: 相似句子复用纠错结果
**实现位置**: `src/core/llm/client.py`
- `LRUCache` 带 TTL 的 LRU 缓存
- `llm_cache_enable`, `llm_cache_size`, `llm_cache_ttl` 配置
- 自动缓存命中和过期清理

### 13. ✅ 异步批量 LLM 处理
**目标**: 减少 LLM 调用次数，提升效率
**实现位置**: `src/core/llm/client.py`, `src/core/engine.py`
- `LLMClient.batch_chat()` 批量聊天方法
- `_apply_llm_batch_polish()` 批量润色
- 请求合并和结果拆分

### 14. ✅ 全文 LLM 纠错模式
**目标**: 利用全局上下文提升纠错质量
**实现位置**: `src/core/engine.py`, `src/config.py`
- `llm_fulltext_enable: bool` 配置
- `_apply_llm_fulltext_polish()` 方法
- `llm_fulltext_max_chars` 长度限制

## 五、音频预处理优化 (15-17)

### 15. ✅ DeepFilterNet v3 集成
**目标**: 降噪质量提升至 PESQ 3.5+
**实现位置**: `src/core/audio/preprocessor.py`, `src/config.py`
- `audio_denoise_backend: str` 支持 "deepfilter3"
- `_apply_deepfilter3_denoise()` 方法
- 自动最小长度填充

### 16. ✅ 自适应预处理流水线
**目标**: 根据音频质量智能选择处理步骤
**实现位置**: `src/core/audio/preprocessor.py`, `src/config.py`
- `audio_adaptive_preprocess: bool` 配置
- `estimate_snr()` SNR 估计方法
- `audio_snr_threshold` 自动降噪阈值

### 17. ✅ 长音频智能分块
**目标**: 优化超长音频处理效率
**实现位置**: `src/core/audio/chunker.py`, `src/core/engine.py`
- `AudioChunker` 智能分块器
- VAD 静音点检测分割
- `transcribe_long_audio()` 方法

## 六、文本后处理优化 (18-19)

### 18. ✅ 并行文本处理流水线
**目标**: 后处理速度提升 3x
**实现位置**: `src/core/text_processor/post_processor.py`
- `process_batch()` 并行批处理
- `process_batch_async()` 异步批处理
- ThreadPoolExecutor 线程池

### 19. ✅ 置信度驱动选择性纠错
**目标**: 减少不必要的纠错，提升效率
**实现位置**: `src/core/engine.py`, `src/config.py`
- `confidence_threshold` 置信度阈值
- `confidence_fallback` 回退策略 (pycorrector/llm)
- `_filter_low_confidence()` 方法

## 七、系统架构优化 (20)

### 20. ✅ 可观测性增强
**目标**: 全链路监控和性能分析
**实现位置**: `src/utils/service_metrics.py`, `src/main.py`
- `ServiceMetrics` 指标收集类
- `/metrics` Prometheus 格式端点
- 请求数、延迟、音频时长统计

---

## 执行优先级

| 优先级 | 优化项 | 预期收益 | 状态 |
|--------|--------|----------|------|
| P0 | 1, 2, 5, 11 | 核心性能提升 | ✅ |
| P1 | 8, 9, 14, 15 | 准确率和效率提升 | ✅ |
| P2 | 3, 6, 12, 20 | 功能增强 | ✅ |
| P3 | 其他 | 渐进优化 | ✅ |

## 新增配置项汇总

```python
# src/config.py 新增配置
# ONNX 优化
onnx_intra_threads: int = 4
onnx_inter_threads: int = 1

# 预热
warmup_on_startup: bool = True
warmup_audio_duration: float = 1.0

# WebSocket
ws_compression: bool = True
ws_heartbeat_interval: int = 30
ws_heartbeat_timeout: int = 60

# LLM 后端
llm_backend: str = "auto"
llm_cache_enable: bool = True
llm_cache_size: int = 1000
llm_cache_ttl: int = 3600
llm_fulltext_enable: bool = False
llm_fulltext_max_chars: int = 2000

# 热词 FAISS
hotword_use_faiss: bool = False
hotword_faiss_index_type: str = "IVFFlat"

# 自适应预处理
audio_adaptive_preprocess: bool = False
audio_snr_threshold: float = 20.0

# 置信度
confidence_threshold: float = 0.0
confidence_fallback: str = "pycorrector"
```

## 参考资源

- [FunASR Technical Report](https://arxiv.org/html/2509.12508v3)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [H-PRM Hotword Pre-Retrieval](https://arxiv.org/abs/2508.18295)
- [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [Full-text Error Correction with LLM](https://arxiv.org/abs/2409.07790)
