# Device-tier model recommendation

TokenSmith classifies a device into the highest local tier that passes the policy. Tier 0 recommends one cloud model when no local tier passes.

## Scope

- Recommendation only; the policy does not change model runtime settings.
- The cloud tier names a hosted model but does not set up, authenticate, or test its provider.
- Five exact local Ollama configurations and one cloud configuration.
- Gemma 3 4B Q4 is the lightweight local entry point; devices that cannot run it safely receive the cloud recommendation.
- Memory estimates assume TokenSmith's existing 2,048-token context.
- macOS Sonoma or newer, Windows 10 22H2 or newer, and Linux are evaluated.
- Intel macOS is evaluated through Ollama's CPU path. Apple Silicon may use unified GPU memory.
- Windows ARM may use the CPU path. Accelerators count only when runtime support is verified.

## Flow

1. Platform-specific detectors produce one `DeviceCapabilities` value.
2. The tier classifier evaluates Tier 1 through Tier 5.
3. The highest eligible tier is selected.
4. The recommendation mapper selects that tier's model.
5. If no local tier is eligible, Tier 0 selects the cloud model.

## Tier policy

| Tier | Local model | CPU requirement | Unified memory | Dedicated VRAM | Free disk |
| --- | --- | --- | --- | --- | --- |
| 1 Light | Gemma 3 4B Q4_K_M | 16 GiB, 8 threads | 8 GiB | 4 GiB | 5 GiB |
| 2 Standard | Gemma 3 4B Q8_0 | 24 GiB, 12 threads | 16 GiB | 8 GiB | 8 GiB |
| 3 Enhanced | Gemma 3 12B Q4_K_M | Not recommended | 20 GiB | 10 GiB | 10 GiB |
| 4 High Precision | Gemma 3 12B Q8_0 | Not recommended | 32 GiB | 16 GiB | 16 GiB |
| 5 Workstation | Gemma 3 27B Q8_0 | Not recommended | 48 GiB | 32 GiB | 35 GiB |
| 0 Cloud | Gemini 2.5 Flash through Google AI Studio | No local requirement | No local requirement | No local requirement | No model download |

A dedicated-GPU path must also meet the tier's host-memory minimum. Shared graphics memory is not added to system memory.

## Evidence categories

### Hard checks

- The OS and architecture are supported by the local Ollama runtime.
- The model download fits in available storage.
- A GPU path is used only when its runtime support is verified.
- Shared memory is counted once.

### Conservative policy heuristics

- Total RAM, unified-memory, and VRAM thresholds.
- CPU thread minimums.
- Selecting the highest tier that fits.
- Disabling CPU-only recommendations for Tiers 3 through 5.
- Treating Gemma 3 4B Q4 as the local entry model for TokenSmith's study workflow.

### Benchmark validation required

- Peak RAM and VRAM while loading and generating.
- Time to first token and output tokens per second.
- Thermal throttling during sustained use.
- Performance on each supported OS and architecture.
- Whether partial GPU offload should affect tier placement.

## Benchmark protocol

Keep the existing model settings unchanged. For each tier and representative device, record:

1. successful model load;
2. peak host memory;
3. peak accelerator memory;
4. time to first token;
5. output tokens per second;
6. completion success for a fixed TokenSmith study prompt;
7. sustained performance after repeated prompts.

The benchmark results should tune the heuristic thresholds without changing the tier-classification architecture.

## Model metadata sources

- Ollama Gemma 3 tags: <https://ollama.com/library/gemma3/tags>
- Ollama macOS requirements: <https://github.com/ollama/ollama/blob/main/docs/macos.mdx>
- Ollama Windows requirements: <https://github.com/ollama/ollama/blob/main/docs/windows.mdx>
- Gemini model catalog: <https://ai.google.dev/gemini-api/docs/models>
