/** Electron's MAS runtime determines the channel; an environment variable cannot relax it. */
export function isMacAppStore(): boolean {
  return process.mas === true
}

export function assertOllamaManagementAllowed(): void {
  if (isMacAppStore()) {
    throw new Error('Manage Ollama and its models outside TokenSmith. You can connect to an already running Ollama server or use a cloud model.')
  }
}
