import apiClient from './client'
import type { HotwordsListResponse, HotwordsUpdateResponse } from './types'

/**
 * 获取当前热词列表
 */
export async function getHotwords(): Promise<HotwordsListResponse> {
  const response = await apiClient.get<HotwordsListResponse>('/api/v1/hotwords')
  return response.data
}

/**
 * 更新热词列表（替换全部）
 */
export async function updateHotwords(hotwords: string[]): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>(
    '/api/v1/hotwords',
    { hotwords }
  )
  return response.data
}

/**
 * 追加热词（保留现有）
 */
export async function appendHotwords(hotwords: string[]): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>(
    '/api/v1/hotwords/append',
    { hotwords }
  )
  return response.data
}

/**
 * 从文件重新加载热词
 */
export async function reloadHotwords(): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>('/api/v1/hotwords/reload')
  return response.data
}

/**
 * 获取当前上下文热词列表（仅用于注入提示，不做强制替换）
 */
export async function getContextHotwords(): Promise<HotwordsListResponse> {
  const response = await apiClient.get<HotwordsListResponse>('/api/v1/hotwords/context')
  return response.data
}

/**
 * 更新上下文热词列表（替换全部）
 */
export async function updateContextHotwords(hotwords: string[]): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>(
    '/api/v1/hotwords/context',
    { hotwords }
  )
  return response.data
}

/**
 * 追加上下文热词（保留现有）
 */
export async function appendContextHotwords(hotwords: string[]): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>(
    '/api/v1/hotwords/context/append',
    { hotwords }
  )
  return response.data
}

/**
 * 从文件重新加载上下文热词
 */
export async function reloadContextHotwords(): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>('/api/v1/hotwords/context/reload')
  return response.data
}
