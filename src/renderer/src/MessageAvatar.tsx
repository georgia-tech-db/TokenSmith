import { CircleUserRound } from 'lucide-react'
import tokensmithAssistantMark from './assets/tokensmith-assistant-mark.png'

export function MessageAvatar({ role }: { role: 'user' | 'assistant' }) {
  return role === 'user'
    ? <CircleUserRound className="message-avatar user" size={26} aria-hidden="true" />
    : <img className="message-avatar assistant assistant-mark" src={tokensmithAssistantMark} alt="" aria-hidden="true" />
}
