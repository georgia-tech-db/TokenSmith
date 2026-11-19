export const libraryNameKey = (name: string) => name.normalize('NFKC').trim().replace(/\s+/g, ' ').toLocaleLowerCase()

export function uniqueLibraryName(base: string, existing: string[]): string {
  const names = new Set(existing.map(libraryNameKey))
  const stem = base.trim().slice(0, 190) || 'Library'
  let name = stem, suffix = 2
  while (names.has(libraryNameKey(name))) name = `${stem} (${suffix++})`
  return name
}
