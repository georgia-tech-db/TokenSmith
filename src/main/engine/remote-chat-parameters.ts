// Use the same request parameters for connection checks and study chat.
export function remoteChatParameters(baseUrl: string, model: string, maxTokens?: number, temperature?: number, topP?: number) {
  const openai = new URL(baseUrl).hostname === 'api.openai.com'
  const reasoning = openai && /^(o\d|gpt-[5-9](?:[.\-]|$))/i.test(model)
  return {
    ...(openai ? { max_completion_tokens: maxTokens } : { max_tokens: maxTokens }),
    ...(!reasoning ? { temperature, top_p: topP } : {})
  }
}
