import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

const questionBlockElements = ['blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'ul', 'ol', 'li', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'input']

export function MessageText({ text, inline = false }: { text: string; inline?: boolean }) {
  const Container = inline ? 'span' : 'div'
  return (
    <Container className={inline ? 'suggestion-text' : 'message-text'}>
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
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
          table: ({ children }) => <div className="message-table"><table>{children}</table></div>
        }}
      >
        {text}
      </Markdown>
    </Container>
  )
}
