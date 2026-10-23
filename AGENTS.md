# AGENTS.md

**The canonical agent guidance lives in [`CLAUDE.md`](CLAUDE.md).** Read it in full before making changes. The docs index is [`docs/README.md`](docs/README.md).

The non-negotiables, in case you read nothing else:
1. **Every LLM call goes through the Mubit gateway** (`https://api.mubit.ai/...`, auth with `MUBIT_API_KEY`). Never call a provider directly.
2. **Stack:** FastAPI + SQLite backend, Vite + React + shadcn/ui + Tailwind frontend, Modal for compute, Pen.dev for design.
3. **No unmeasured numbers** in the UI, README or slides.
4. **Don't change agent prompts casually.** Prompt drift splits Mubit tasks and resets adapter training.
5. **Commit or push only when the user asks.**
