import { execFile } from 'node:child_process'

export function runDetectorCommand(
  command: string,
  args: string[],
  timeout = 3000
): Promise<string | null> {
  return new Promise((resolve) => {
    execFile(
      command,
      args,
      {
        encoding: 'utf8',
        timeout,
        windowsHide: true,
        maxBuffer: 1024 * 1024
      },
      (error, stdout) => {
        resolve(error ? null : stdout.trim())
      }
    )
  })
}
