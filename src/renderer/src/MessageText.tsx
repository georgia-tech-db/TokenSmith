import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { remarkModelMath } from './remark-model-math'

/** Must match `citationHrefPrefix` in citations.ts. */
const citationHrefPrefix = '#cite-'

const questionBlockElements = ['blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'ul', 'ol', 'li', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'input']

export function MessageText({
  text,
  inline = false,
  onCite
}: {
  text: string
  inline?: boolean
  /** Opens the source behind citation marker `claimIndex`. */
  onCite?: (claimIndex: number) => void
}) {
  const Container = inline ? 'span' : 'div'
  return (
    <Container className={inline ? 'suggestion-text' : 'message-text'}>
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath, remarkModelMath]}
        rehypePlugins={[[rehypeKatex, { trust: false, maxSize: 10 }]]}
        disallowedElements={inline ? questionBlockElements : undefined}
        unwrapDisallowed={inline}
        urlTransform={(url) =>
          /^(https?:\/\/|mailto:)/i.test(url) || (onCite && url.startsWith(citationHrefPrefix)) ? url : undefined}
        components={{
          p: ({ children }) => inline ? <span>{children}</span> : <p>{children}</p>,
          div: ({ children, className }) => inline
            ? <span className={className}>{children}</span>
            : <div className={className}>{children}</div>,
          a: ({ href, children }) => {
            if (href && onCite && href.startsWith(citationHrefPrefix)) {
              const claimIndex = Number(href.slice(citationHrefPrefix.length))
              return (
                <button
                  className="citation-marker"
                  type="button"
                  aria-label={`Open source ${String(children)}`}
                  onClick={() => onCite(claimIndex)}
                >
                  {children}
                </button>
              )
            }

            return href && !inline
              ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
              : <span>{children}</span>
          },
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
