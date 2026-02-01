import { useState } from 'react'
import { toast } from 'sonner'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Loader2, Play } from 'lucide-react'
import { FileDropzone } from '@/components/upload'
import { TranscribeOptions } from '@/components/transcribe'
import { TranscriptView } from '@/components/transcript'
import { Timeline } from '@/components/timeline'
import { useTranscriptionStore } from '@/stores'
import { transcribeAudio, transcribeBatch } from '@/lib/api'
import type { SentenceInfo } from '@/lib/api/types'

export default function TranscribePage() {
  const {
    files,
    options,
    tempHotwords,
    result,
    setResult,
    isTranscribing,
    setTranscribing,
    clearFiles,
    setSelectedSentence,
  } = useTranscriptionStore()

  const [selectedIndex, setSelectedIndex] = useState<number>()

  const handleTranscribe = async () => {
    if (files.length === 0) {
      toast.error('请先上传音频文件')
      return
    }

    setTranscribing(true)
    setResult(null)

    try {
      const transcribeOptions = {
        ...options,
        hotwords: tempHotwords || undefined,
      }

      if (files.length === 1) {
        // 单文件转写
        const response = await transcribeAudio(files[0], transcribeOptions)
        if (response.code === 0) {
          setResult(response)
          toast.success('转写完成')
        } else {
          toast.error('转写失败')
        }
      } else {
        // 批量转写
        const response = await transcribeBatch(files, transcribeOptions)
        if (response.success_count > 0) {
          // 显示第一个成功的结果
          const firstSuccess = response.results.find(r => r.success && r.result)
          if (firstSuccess?.result) {
            setResult(firstSuccess.result)
          }
          toast.success(`转写完成: ${response.success_count}/${response.total} 成功`)
        } else {
          toast.error('所有文件转写失败')
        }
      }
    } catch (error) {
      console.error('Transcription error:', error)
      toast.error('转写请求失败，请检查服务连接')
    } finally {
      setTranscribing(false)
    }
  }

  const handleSelectSentence = (sentence: SentenceInfo, index: number) => {
    setSelectedSentence(sentence)
    setSelectedIndex(index)
  }

  const handleClear = () => {
    clearFiles()
    setResult(null)
    setSelectedSentence(null)
    setSelectedIndex(undefined)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">音频转写</h1>
        <p className="text-muted-foreground">上传音频文件进行语音识别和转写</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* 上传区域 */}
        <Card>
          <CardHeader>
            <CardTitle>上传文件</CardTitle>
            <CardDescription>支持 WAV, MP3, M4A, FLAC, OGG, MP4 等格式</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <FileDropzone />

            <div className="flex gap-2">
              <Button
                className="flex-1"
                onClick={handleTranscribe}
                disabled={files.length === 0 || isTranscribing}
              >
                {isTranscribing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    转写中...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    开始转写
                  </>
                )}
              </Button>
              {files.length > 0 && !isTranscribing && (
                <Button variant="outline" onClick={handleClear}>
                  清空
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* 转写选项 */}
        <TranscribeOptions />
      </div>

      {/* 时间轴 */}
      {result && result.sentences.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">时间轴</CardTitle>
          </CardHeader>
          <CardContent>
            <Timeline
              sentences={result.sentences}
              selectedIndex={selectedIndex}
              onSelectSentence={handleSelectSentence}
            />
          </CardContent>
        </Card>
      )}

      {/* 转写结果 */}
      {result && (
        <TranscriptView
          result={result}
          filename={files[0]?.name?.replace(/\.[^/.]+$/, '')}
        />
      )}

      {/* 空状态 */}
      {!result && !isTranscribing && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="text-muted-foreground text-center">
              上传音频文件并点击"开始转写"按钮，转写结果将显示在这里
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
