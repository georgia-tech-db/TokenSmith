// The desktop installs Chromium's transport after app.ready so cloud generators
// honor system proxies. Keep the Node default for standalone engine/unit tests.
let transport: typeof fetch | undefined

export function setRemoteGeneratorTransport(request: typeof fetch): void { transport = request }

export const remoteGeneratorFetch: typeof fetch = (input, init) => (transport ?? fetch)(input, init)
