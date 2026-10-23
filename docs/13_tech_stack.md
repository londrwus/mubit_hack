# Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | **Python + FastAPI** | REST + WebSocket for streaming live race metrics |
| Database | **SQLite** | Runs, experiments, corrections, results. Move to PostgreSQL only if there is a real need. |
| Frontend | **Vite + React** | The race dashboard (see 09_demo_showcase.md) |
| UI components | **shadcn/ui + Tailwind CSS** | Dark theme, large type for the stage |
| Design | **Pen.dev** | Dashboard mockups before building |
| Compute | **Pluggable runners** (`local`, `modal`) | The product is compute-agnostic; **Modal is only test/demo compute** ($30 budget, see 08_modal_compute_budget.md) |
| LLM routing | **Mubit gateway** | Every LLM call goes through `api.mubit.ai` (hackathon rule) |

See `architecture.md` for how these fit together.
