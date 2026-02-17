import apiClient from './client'
import type { HealthResponse, MetricsResponse } from './types'

/**
 * 健康检查
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}

/**
 * 获取服务指标
 */
export async function getMetrics(): Promise<MetricsResponse> {
  const response = await apiClient.get<MetricsResponse>('/metrics')
  return response.data
}

/**
 * 获取 Prometheus 文本指标
 */
export async function getPrometheusMetrics(): Promise<string> {
  const response = await apiClient.get<string>('/metrics/prometheus', {
    responseType: 'text',
  })
  return response.data
}
