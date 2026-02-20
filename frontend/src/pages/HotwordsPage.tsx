import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Loader2, RefreshCw, Plus, Upload, Search } from 'lucide-react'
import {
  appendContextHotwords,
  appendHotwords,
  getContextHotwords,
  getHotwords,
  reloadContextHotwords,
  reloadHotwords,
  updateContextHotwords,
  updateHotwords,
} from '@/lib/api'
import { useBackendStore } from '@/stores'

export default function HotwordsPage() {
  const queryClient = useQueryClient()
  const { baseUrl } = useBackendStore()
  const [mode, setMode] = useState<'forced' | 'context'>('forced')
  const [draftText, setDraftText] = useState('')
  const [isDirty, setIsDirty] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')

  const isContextMode = mode === 'context'

  // 获取热词列表
  const { data, isLoading, refetch } = useQuery({
    queryKey: ['hotwords', mode, baseUrl],
    queryFn: isContextMode ? getContextHotwords : getHotwords,
  })

  const serverText = (data?.hotwords ?? []).join('\n')
  const editText = isDirty ? draftText : serverText

  // 更新热词
  const updateMutation = useMutation({
    mutationFn: (hotwords: string[]) =>
      isContextMode ? updateContextHotwords(hotwords) : updateHotwords(hotwords),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['hotwords'] })
      setIsDirty(false)
      setDraftText('')
      toast.success(`${isContextMode ? '上下文热词' : '热词'}更新成功，共 ${response.count} 个`)
    },
    onError: () => {
      toast.error(`${isContextMode ? '上下文热词' : '热词'}更新失败`)
    },
  })

  // 追加热词
  const appendMutation = useMutation({
    mutationFn: (hotwords: string[]) =>
      isContextMode ? appendContextHotwords(hotwords) : appendHotwords(hotwords),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['hotwords'] })
      setIsDirty(false)
      setDraftText('')
      toast.success(response.message)
    },
    onError: () => {
      toast.error(`追加${isContextMode ? '上下文热词' : '热词'}失败`)
    },
  })

  // 重载热词
  const reloadMutation = useMutation({
    mutationFn: isContextMode ? reloadContextHotwords : reloadHotwords,
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['hotwords'] })
      setIsDirty(false)
      setDraftText('')
      toast.success(response.message)
    },
    onError: () => {
      toast.error(`重载${isContextMode ? '上下文热词' : '热词'}失败`)
    },
  })

  // 解析编辑框内容为热词数组
  const parseHotwords = (text: string): string[] => {
    return text
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'))
  }

  // 更新全部
  const handleUpdate = () => {
    const hotwords = parseHotwords(editText)
    if (hotwords.length === 0) {
      toast.warning('请输入至少一个热词')
      return
    }
    updateMutation.mutate(hotwords)
  }

  // 追加热词
  const handleAppend = () => {
    const hotwords = parseHotwords(editText)
    const existingSet = new Set(data?.hotwords || [])
    const newHotwords = hotwords.filter((hw) => !existingSet.has(hw))

    if (newHotwords.length === 0) {
      toast.info('没有新的热词需要追加')
      return
    }
    appendMutation.mutate(newHotwords)
  }

  // 重载热词
  const handleReload = () => {
    reloadMutation.mutate()
  }

  // 过滤显示的热词
  const filteredHotwords = data?.hotwords.filter((hw) =>
    hw.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const isPending = updateMutation.isPending || appendMutation.isPending || reloadMutation.isPending

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">热词管理</h1>
        <p className="text-muted-foreground">
          管理转写热词，提高专业术语识别准确率（强制纠错 / 上下文注入）
        </p>
      </div>

      <Tabs
        value={mode}
        onValueChange={(v) => {
          const next = (v === 'context' ? 'context' : 'forced') as typeof mode
          if (next !== mode) {
            setMode(next)
            setIsDirty(false)
            setDraftText('')
            setSearchTerm('')
          }
        }}
      >
        <TabsList>
          <TabsTrigger value="forced">强制热词（纠错）</TabsTrigger>
          <TabsTrigger value="context">上下文热词（注入提示）</TabsTrigger>
        </TabsList>

        <TabsContent value="forced">
          <p className="text-sm text-muted-foreground">
            强制热词会进入纠错链路（可能替换相似词）；适合少量高确定性术语。
          </p>
        </TabsContent>
        <TabsContent value="context">
          <p className="text-sm text-muted-foreground">
            上下文热词仅用于提示/注入（不会强制替换）；更适合会议专有名词清单。
          </p>
        </TabsContent>
      </Tabs>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* 热词编辑 */}
        <Card>
          <CardHeader>
            <CardTitle>{isContextMode ? '上下文热词编辑' : '强制热词编辑'}</CardTitle>
            <CardDescription>
              每行一个热词，以 # 开头的行为注释
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder={
                isContextMode
                  ? "输入上下文热词，每行一个...&#10;# 这是注释&#10;Qwen3-ASR&#10;VibeVoice-ASR&#10;Kubernetes"
                  : "输入热词，每行一个...&#10;# 这是注释&#10;麦当劳&#10;肯德基&#10;Bilibili"
              }
              value={editText}
              onChange={(e) => {
                if (!isDirty) setIsDirty(true)
                setDraftText(e.target.value)
              }}
              className="min-h-[300px] font-mono text-sm"
            />
            <div className="flex flex-wrap gap-2">
              <Button onClick={handleUpdate} disabled={isPending}>
                {updateMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                更新全部
              </Button>
              <Button variant="outline" onClick={handleAppend} disabled={isPending}>
                {appendMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                <Plus className="h-4 w-4 mr-2" />
                追加{isContextMode ? '上下文热词' : '热词'}
              </Button>
              <Button variant="outline" onClick={handleReload} disabled={isPending}>
                {reloadMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                <Upload className="h-4 w-4 mr-2" />
                从文件重载
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 当前热词 */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>当前{isContextMode ? '上下文热词库' : '热词库'}</CardTitle>
                <CardDescription>
                  共 <Badge variant="secondary">{data?.count ?? 0}</Badge> 个{isContextMode ? '上下文热词' : '热词'}
                </CardDescription>
              </div>
              <Button variant="ghost" size="icon" onClick={() => refetch()}>
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 搜索 */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder={`搜索${isContextMode ? '上下文热词' : '热词'}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>

            {/* 热词列表 */}
            <div className="h-[280px] overflow-y-auto rounded-lg border p-3">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : filteredHotwords && filteredHotwords.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {filteredHotwords.map((hw, index) => (
                    <Badge key={index} variant="outline">
                      {hw}
                    </Badge>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  {searchTerm ? '没有匹配的热词' : '暂无热词'}
                </div>
              )}
            </div>

            {/* 统计信息 */}
            {searchTerm && filteredHotwords && (
              <p className="text-sm text-muted-foreground text-right">
                显示 {filteredHotwords.length} / {data?.count} 个
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
