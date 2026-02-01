import { useState, useRef, useCallback, useEffect } from 'react'
import type { WSMessage } from '@/lib/api/types'

interface UseWebSocketOptions {
  onMessage?: (message: WSMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  isConnecting: boolean
  connectionId: string | null
  connect: () => void
  disconnect: () => void
  send: (data: string | ArrayBuffer) => void
  sendJson: (data: object) => void
}

export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnectAttempts = 3,
    reconnectInterval = 2000,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [connectionId, setConnectionId] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setIsConnecting(true)

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        setIsConnecting(false)
        reconnectCountRef.current = 0
        onOpen?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WSMessage

          // 处理连接确认消息
          if ('type' in data && data.type === 'connected') {
            setConnectionId(data.connection_id)
          }

          // 处理心跳
          if ('type' in data && data.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }))
          }

          onMessage?.(data)
        } catch {
          // 非 JSON 消息，忽略
        }
      }

      ws.onclose = () => {
        setIsConnected(false)
        setIsConnecting(false)
        setConnectionId(null)
        onClose?.()

        // 尝试重连
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectCountRef.current++
            connect()
          }, reconnectInterval)
        }
      }

      ws.onerror = (error) => {
        setIsConnecting(false)
        onError?.(error)
      }
    } catch (error) {
      setIsConnecting(false)
      console.error('WebSocket connection error:', error)
    }
  }, [url, onMessage, onOpen, onClose, onError, reconnectAttempts, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    reconnectCountRef.current = reconnectAttempts // 防止重连
    wsRef.current?.close()
    wsRef.current = null
    setIsConnected(false)
    setConnectionId(null)
  }, [reconnectAttempts])

  const send = useCallback((data: string | ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data)
    }
  }, [])

  const sendJson = useCallback((data: object) => {
    send(JSON.stringify(data))
  }, [send])

  // 清理
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      wsRef.current?.close()
    }
  }, [])

  return {
    isConnected,
    isConnecting,
    connectionId,
    connect,
    disconnect,
    send,
    sendJson,
  }
}
