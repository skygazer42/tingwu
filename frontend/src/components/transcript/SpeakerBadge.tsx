import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const SPEAKER_COLORS = [
  { bg: 'bg-blue-500/10', text: 'text-blue-600', border: 'border-blue-200' },
  { bg: 'bg-green-500/10', text: 'text-green-600', border: 'border-green-200' },
  { bg: 'bg-purple-500/10', text: 'text-purple-600', border: 'border-purple-200' },
  { bg: 'bg-orange-500/10', text: 'text-orange-600', border: 'border-orange-200' },
  { bg: 'bg-pink-500/10', text: 'text-pink-600', border: 'border-pink-200' },
  { bg: 'bg-cyan-500/10', text: 'text-cyan-600', border: 'border-cyan-200' },
  { bg: 'bg-amber-500/10', text: 'text-amber-600', border: 'border-amber-200' },
  { bg: 'bg-indigo-500/10', text: 'text-indigo-600', border: 'border-indigo-200' },
]

interface SpeakerBadgeProps {
  speaker?: string
  speakerId?: number
  className?: string
}

export function SpeakerBadge({ speaker, speakerId, className }: SpeakerBadgeProps) {
  if (!speaker && speakerId === undefined) {
    return null
  }

  const colorIndex = speakerId !== undefined ? speakerId % SPEAKER_COLORS.length : 0
  const colors = SPEAKER_COLORS[colorIndex]
  const label = speaker || `说话人 ${String.fromCharCode(65 + (speakerId ?? 0))}`

  return (
    <Badge
      variant="outline"
      className={cn(
        colors.bg,
        colors.text,
        colors.border,
        'font-medium',
        className
      )}
    >
      {label}
    </Badge>
  )
}

export function getSpeakerColor(speakerId: number) {
  return SPEAKER_COLORS[speakerId % SPEAKER_COLORS.length]
}
