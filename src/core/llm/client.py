"""LLM 客户端 - 支持 Ollama 和 OpenAI 兼容接口

提供统一的 LLM 调用接口，支持流式输出。
"""
import httpx
from typing import AsyncGenerator, List, Dict, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class LLMMessage:
    """LLM消息"""
    role: str       # "system", "user", "assistant"
    content: str


class LLMClient:
    """
    LLM 客户端

    支持:
    - Ollama (本地)
    - OpenAI 兼容接口
    - 流式输出
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0
    ):
        """
        初始化 LLM 客户端

        Args:
            base_url: API 基础 URL
            model: 模型名称
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
            timeout: 请求超时时间(秒)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 判断是 Ollama 还是 OpenAI 风格
        self.is_ollama = "11434" in base_url or "ollama" in base_url.lower()

    async def chat(
        self,
        messages: List[LLMMessage],
        stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            stream: 是否流式输出

        Yields:
            str: 生成的文本片段（如果stream=True）或完整响应（stream=False）
        """
        if self.is_ollama:
            async for chunk in self._chat_ollama(messages, stream):
                yield chunk
        else:
            async for chunk in self._chat_openai(messages, stream):
                yield chunk

    async def _chat_ollama(
        self,
        messages: List[LLMMessage],
        stream: bool
    ) -> AsyncGenerator[str, None]:
        """Ollama API 调用"""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                if stream:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    content = await response.aread()
                    data = json.loads(content)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

    async def _chat_openai(
        self,
        messages: List[LLMMessage],
        stream: bool
    ) -> AsyncGenerator[str, None]:
        """OpenAI 兼容 API 调用"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                if stream:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line_data = line[6:]
                            if line_data.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(line_data)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                else:
                    content = await response.aread()
                    data = json.loads(content)
                    if "choices" in data and len(data["choices"]) > 0:
                        yield data["choices"][0]["message"]["content"]


async def test_client():
    """测试客户端"""
    client = LLMClient()
    messages = [
        LLMMessage(role="system", content="你是一个智能助手"),
        LLMMessage(role="user", content="你好，请介绍一下自己")
    ]

    print("=== 非流式输出 ===")
    async for response in client.chat(messages, stream=False):
        print(response)

    print("\n=== 流式输出 ===")
    async for chunk in client.chat(messages, stream=True):
        print(chunk, end='', flush=True)
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_client())
