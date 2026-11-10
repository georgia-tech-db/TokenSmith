import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { remarkModelMath } from './remark-model-math'

const questionBlockElements = ['blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'ul', 'ol', 'li', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'input']

function alignCodeLineNumbers(text: string): string {
  const lines = text.split('\n')
  if (lines.at(-1) === '') lines.pop()
  const numbered = lines.map(line => /^[ \t]*(\d+)\t(.*)$/.exec(line))
  if (numbered.length < 2 || numbered.some((line, index) => !line ||
    (index > 0 && Number(line[1]) !== Number(numbered[index - 1]?.[1]) + 1))) return text
  const width = Math.max(...numbered.map(line => line![1].length))
  // The tab separates the printed line number from code; preserve everything after it.
  return numbered.map(line => line![1].padStart(width) + '\t' + line![2]).join('\n') + (text.endsWith('\n') ? '\n' : '')
}

export function MessageText({ text, inline = false }: { text: string; inline?: boolean }) {
  const Container = inline ? 'span' : 'div'
  return (
    <Container className={inline ? 'suggestion-text' : 'message-text'}>
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath, remarkModelMath]}
        rehypePlugins={[[rehypeKatex, { trust: false, maxSize: 10 }]]}
        disallowedElements={inline ? questionBlockElements : undefined}
        unwrapDisallowed={inline}
        urlTransform={(url) => /^(https?:\/\/|mailto:)/i.test(url) ? url : undefined}
        components={{
          p: ({ children }) => inline ? <span>{children}</span> : <p>{children}</p>,
          div: ({ children, className }) => inline
            ? <span className={className}>{children}</span>
            : <div className={className}>{children}</div>,
          a: ({ href, children }) => href && !inline
            ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
            : <span>{children}</span>,
          // Model output must not load remote images or local files.
          img: ({ alt }) => <span>{alt}</span>,
          code: ({ children, className }) => <code className={className}>{
            typeof children === 'string' && children.includes('\n') ? alignCodeLineNumbers(children) : children
          }</code>,
          table: ({ children }) => <div className="message-table"><table>{children}</table></div>
        }}
      >
        {text}
      </Markdown>
    </Container>
  )
}
