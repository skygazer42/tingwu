import apiClient from './client'
import type { TranscribeResponse, BatchTranscribeResponse, TranscribeOptions } from './types'

/**
 * 单文件转写
 */
export async function transcribeAudio(
  file: File,
  options: TranscribeOptions = {}
): Promise<TranscribeResponse> {
  const formData = new FormData()
  formData.append('file', file)

  if (options.with_speaker !== undefined) {
    formData.append('with_speaker', String(options.with_speaker))
  }
  if (options.apply_hotword !== undefined) {
    formData.append('apply_hotword', String(options.apply_hotword))
  }
  if (options.apply_llm !== undefined) {
    formData.append('apply_llm', String(options.apply_llm))
  }
  if (options.llm_role) {
    formData.append('llm_role', options.llm_role)
  }
  if (options.hotwords) {
    formData.append('hotwords', options.hotwords)
  }

  const response = await apiClient.post<TranscribeResponse>(
    '/api/v1/transcribe',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

/**
 * 批量文件转写
 */
export async function transcribeBatch(
  files: File[],
  options: TranscribeOptions & { max_concurrent?: number } = {}
): Promise<BatchTranscribeResponse> {
  const formData = new FormData()

  files.forEach((file) => {
    formData.append('files', file)
  })

  if (options.with_speaker !== undefined) {
    formData.append('with_speaker', String(options.with_speaker))
  }
  if (options.apply_hotword !== undefined) {
    formData.append('apply_hotword', String(options.apply_hotword))
  }
  if (options.apply_llm !== undefined) {
    formData.append('apply_llm', String(options.apply_llm))
  }
  if (options.llm_role) {
    formData.append('llm_role', options.llm_role)
  }
  if (options.hotwords) {
    formData.append('hotwords', options.hotwords)
  }
  if (options.max_concurrent !== undefined) {
    formData.append('max_concurrent', String(options.max_concurrent))
  }

  const response = await apiClient.post<BatchTranscribeResponse>(
    '/api/v1/transcribe/batch',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}
