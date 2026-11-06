# Can My PC Run It? — Local AI Hardware Checker

**Free, no sign-up, no data collected.** Find out which local AI models (LLMs) your PC can run, how fast they'll answer, and how to run bigger ones.

## What this is

- **Browser checker** (`index.html`): enter your RAM, GPU VRAM, and GPU class (or click *Detect my hardware*). It instantly shows which of 13 models will fit, the best quantization, estimated speed with a ±40% range, and a copyable `ollama run` command for every recommendation. Everything runs in your browser — nothing is uploaded.
- **Probe** (`llm_probe.py`): a single readable Python script (standard library only, no installs, no network) that detects your real CPU, RAM, and GPU, measures memory-copy and disk speed, then prints a diagnosis and prescription. Run `python llm_probe.py`. Flags: `--vram GB`, `--ram GB`, `--json`, `--cleanup` (deletes the script itself).
- **Model catalog** (`models.json`): the shared list of 13 models (name, size, Ollama tag, MoE active params), quantization table, and last-updated date. Edit this one file to update models — both the website and the probe read it.

## Speeds are estimates

Estimated using physics: memory capacity decides what fits, memory bandwidth decides speed (each generated word reads the active weights once). Every speed is labeled an **estimate** (±~40%). Real speed depends on inference software, drivers, cooling, background apps, and context length. Always run a model and time it to be sure.

## Calibration (Phase 1 results)

Tested on: Windows 11, AMD CPU 12 threads, 16 GB RAM, AMD Radeon integrated graphics.

| Model | Estimate | Real (Ollama) | Within range? |
|-------|----------|---------------|---------------|
| llama3.2:3b | 8.5 toks/s (5–12) | 5.9 toks/s | ✓ Yes |

Real/estimate ratio ≈ 0.67 → efficiency factor calibrated to 0.68. More hardware results welcome via issues/PRs.

## Folder contents

- `index.html` — the site (self-contained, fonts with system fallbacks)
- `llm_probe.py` — the probe (works standalone; loads `models.json` if present)
- `models.json` — shared model catalog
- `README.md` — this file
- `LICENSE` — MIT

## Editing `models.json`

One entry per model:

```json
{"name": "Llama 3.1 8B", "params": 8.0, "tag": "llama3.1:8b", "moeActive": null, "note": ""}
```

- `params` — total parameters in billions
- `tag` — Ollama tag used in the run command
- `moeActive` — active parameters for Mixture-of-Experts models (e.g. 3.6 for gpt-oss:20b), else `null`
- Update `last_updated` when you change the catalog.

## Deployment (free, static)

**Netlify Drop (easiest):** create a free account at netlify.com, open app.netlify.com/drop, drag the whole `llm-site` folder onto the box. Rename the site in Site settings (e.g. `can-my-pc-run-ai.netlify.app`). To update, drag the new folder onto the Deploys tab.

**GitHub Pages (portfolio route):** create a public repo, upload all files, then Settings → Pages → Deploy from branch → main. Live at `username.github.io/repo-name`.

## Privacy & affiliation

No analytics that identify users, no uploads, no tracking. Model names are used only to identify models. This project is not affiliated with any model maker (Meta, Google, Qwen/Alibaba, Microsoft, OpenAI, etc.).

## License

MIT — see `LICENSE`.
