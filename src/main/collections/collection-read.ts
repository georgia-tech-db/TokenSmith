/** Retry catalog reads when preparation changes their revision in flight. */
export async function readCurrentCollection<T>(collection: { indexRevision: string }, current: () => void,
  read: () => Promise<T>, attempts = 1): Promise<T> {
  for (let attempt = 0; attempt < attempts; attempt++) {
    current()
    const revision = collection.indexRevision
    const result = await read()
    current()
    if (collection.indexRevision === revision) return result
  }
  throw new Error('The library is updating. Please try again shortly.')
}
