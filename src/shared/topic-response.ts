/** One contract for generation and validation of chapter topics. */
interface Schema { type: string; properties?: Record<string, Schema>; required?: string[]; items?: Schema; minItems?: number; maxItems?: number; minLength?: number; maxLength?: number; pattern?: string; enum?: string[]; additionalProperties?: boolean }
const title = { type: 'string', minLength: 1, maxLength: 100, pattern: '^[^\\r\\n?]*[^\\s?][^\\r\\n?]*$' }
const alias = { type: 'string', maxLength: 100 }
const objective = { type: 'string', minLength: 1, maxLength: 500, pattern: '\\S' }
const array = (items: Schema, minItems: number, maxItems: number) => ({ type: 'array', items, minItems, maxItems })
export function topicResponseSchema(allowedIds?: string[]) {
  return { type: 'object', additionalProperties: false, required: ['topics'], properties: {
    topics: array({ type: 'object', additionalProperties: false, required: ['title', 'aliases', 'objectives', 'sourceKeys'], properties: {
      title, aliases: array(alias, 0, 6), objectives: array(objective, 2, 5),
      sourceKeys: array({ type: 'string', ...(allowedIds ? { enum: allowedIds } : {}) }, 1, 6)
    } }, 1, 20)
  } }
}
export interface TopicResponse { topics: { title: string; aliases: string[]; objectives: string[]; sourceKeys: string[] }[] }
// Validate the same small JSON-schema subset sent to the model. Never repair or invent evidence.
export function validateTopicResponse(value: unknown, allowedIds: string[]): asserts value is TopicResponse {
  function check(value: unknown, schema: Schema, path: string): void {
    const fail = (reason: string): never => { throw new Error(`Invalid topic response at ${path}: ${reason}`) }
    if (schema.type === 'object') {
      if (!value || typeof value !== 'object' || Array.isArray(value)) fail('expected an object.')
      const record = value as Record<string, unknown>
      for (const key of Object.keys(record)) if (!(Object.prototype.hasOwnProperty.call(schema.properties, key))) fail(`unexpected field ${key}.`)
      for (const key of schema.required!) check(record[key], schema.properties![key], `${path}.${key}`)
    } else if (schema.type === 'array') {
      if (!Array.isArray(value)) fail('expected an array.')
      const items = value as unknown[]
      if (items.length < schema.minItems! || items.length > schema.maxItems!) fail(`expected ${schema.minItems}–${schema.maxItems} items; received ${items.length}.`)
      items.forEach((item, index) => check(item, schema.items!, `${path}[${index}]`))
    } else {
      if (typeof value !== 'string') fail('expected text.')
      const text = value as string
      const length = Array.from(text).length
      if (schema.maxLength !== undefined && length > schema.maxLength) fail(`maximum ${schema.maxLength} characters; received ${length}.`)
      if (schema.minLength !== undefined && length < schema.minLength) fail('text must not be empty.')
      if (schema.pattern && !new RegExp(schema.pattern).test(text)) fail(path.endsWith('.title') ? 'use a nonempty title without line breaks or question marks.' : 'text must not be blank.')
      if (schema.enum && !schema.enum.includes(text)) fail('source references must use a supplied source key.')
    }
  }
  check(value, topicResponseSchema(allowedIds), 'response')
}
