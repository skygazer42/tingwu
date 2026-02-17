"use client"

import * as React from "react"
import { Link2, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { FormField } from "@/components/ui/form-field"
import { Alert, AlertDescription } from "@/components/ui/alert"

export interface UrlTranscribeProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "onSubmit"> {
  /** 提交回调 */
  onSubmit?: (url: string) => void
  /** 是否加载中 */
  isLoading?: boolean
  /** 禁用状态 */
  disabled?: boolean
}

function UrlTranscribe({
  className,
  onSubmit,
  isLoading = false,
  disabled = false,
  ...props
}: UrlTranscribeProps) {
  const [url, setUrl] = React.useState("")
  const [error, setError] = React.useState("")

  const validateUrl = (value: string): boolean => {
    try {
      const urlObj = new URL(value)
      if (!["http:", "https:"].includes(urlObj.protocol)) {
        setError("仅支持 HTTP 和 HTTPS 协议")
        return false
      }
      // 检查是否是音频/视频文件
      const audioExts = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"]
      const videoExts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
      const path = urlObj.pathname.toLowerCase()
      const hasMediaExt = [...audioExts, ...videoExts].some((ext) => path.endsWith(ext))

      if (!hasMediaExt) {
        setError("URL 应指向音频或视频文件")
        return false
      }
      setError("")
      return true
    } catch {
      setError("请输入有效的 URL")
      return false
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim()) {
      setError("请输入 URL")
      return
    }
    if (!validateUrl(url)) {
      return
    }
    onSubmit?.(url)
  }

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setUrl(value)
    if (value) {
      validateUrl(value)
    } else {
      setError("")
    }
  }

  return (
    <div className={cn("space-y-4", className)} {...props}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <FormField label="音频/视频 URL" required error={error}>
          <div className="relative">
            <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="url"
              placeholder="https://example.com/audio.mp3"
              value={url}
              onChange={handleUrlChange}
              disabled={disabled || isLoading}
              className="pl-10"
            />
          </div>
        </FormField>

        <Button type="submit" disabled={disabled || isLoading || !url.trim()}>
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              提交中...
            </>
          ) : (
            "开始转写"
          )}
        </Button>
      </form>

      <Alert variant="info">
        <AlertDescription>
          URL 转写为异步任务，提交后可在任务队列中查看进度和结果。转写选项（说话人/热词/LLM/高级参数）请在右侧设置。
        </AlertDescription>
      </Alert>
    </div>
  )
}

export { UrlTranscribe }
