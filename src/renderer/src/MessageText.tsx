import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export function MessageText({ text }: { text: string }) {
  return (
    <div className="message-text">
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { trust: false, maxSize: 10 }]]}
        urlTransform={(url) => /^(https?:\/\/|mailto:)/i.test(url) ? url : undefined}
        components={{
          a: ({ href, children }) => href
            ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
            : <span>{children}</span>,
          // Model output must not load remote images or local files.
          img: ({ alt }) => <span>{alt}</span>,
          table: ({ children }) => <div className="message-table"><table>{children}</table></div>
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}
