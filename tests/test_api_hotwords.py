import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    import src.core.engine as engine_mod

    with patch.object(engine_mod, "model_manager") as mock_mm:
        mock_loader = MagicMock()
        mock_mm.loader = mock_loader

        from src.main import app
        with TestClient(app) as c:
            yield c

def test_get_hotwords(client):
    """测试获取热词列表"""
    response = client.get("/api/v1/hotwords")
    assert response.status_code == 200
    data = response.json()
    assert "hotwords" in data
    assert "count" in data

def test_update_hotwords(client):
    """测试更新热词"""
    response = client.post(
        "/api/v1/hotwords",
        json={"hotwords": ["Claude", "Bilibili", "麦当劳"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["count"] == 3

def test_append_hotwords(client):
    """测试追加热词"""
    # 先设置一些热词
    client.post("/api/v1/hotwords", json={"hotwords": ["Claude"]})

    # 追加新热词
    response = client.post(
        "/api/v1/hotwords/append",
        json={"hotwords": ["FunASR", "Python"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["count"] >= 2  # 至少有追加的两个

def test_reload_hotwords(client):
    """测试重新加载热词"""
    response = client.post("/api/v1/hotwords/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0

def test_update_hotwords_empty(client):
    """测试更新空热词列表"""
    response = client.post(
        "/api/v1/hotwords",
        json={"hotwords": []}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0


def test_get_context_hotwords(client):
    """测试获取上下文热词列表"""
    response = client.get("/api/v1/hotwords/context")
    assert response.status_code == 200
    data = response.json()
    assert "hotwords" in data
    assert "count" in data


def test_update_context_hotwords(client):
    """测试更新上下文热词 (替换全部)"""
    response = client.post(
        "/api/v1/hotwords/context",
        json={"hotwords": ["Qwen3-ASR", "VibeVoice-ASR", "GPT4"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["count"] == 3


def test_append_context_hotwords(client):
    """测试追加上下文热词"""
    client.post("/api/v1/hotwords/context", json={"hotwords": ["Claude"]})

    response = client.post(
        "/api/v1/hotwords/context/append",
        json={"hotwords": ["FunASR", "Python"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
    assert data["count"] >= 2


def test_reload_context_hotwords(client):
    """测试从文件重新加载上下文热词"""
    response = client.post("/api/v1/hotwords/context/reload")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == 0
