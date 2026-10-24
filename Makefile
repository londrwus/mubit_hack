# Thin wrappers around scripts/ (Windows without make: run the commands directly).
UV ?= uv

.PHONY: install test dev mock-race probe bench modal-deploy smoke-rl

install:            ## backend + ML libs (CPU torch, jax) + dev tools
	$(UV) sync --extra ml

test:
	$(UV) run --extra ml pytest -q

dev:                ## FastAPI dev server on :8000 (mock LLM + local runner by default)
	$(UV) run --extra ml python scripts/dev_server.py

mock-race:          ## end-to-end race through the API, mock LLM, laptop CPU, $0
	$(UV) run --extra ml python scripts/mock_race.py -v

probe:              ## Opus vs Haiku latency through Mubit, record off (~$0.07)
	$(UV) run python scripts/probe_latency.py --n 3 --max-usd 0.30

bench:              ## list presets, goals and record modes (race: scripts/bench.py run <preset> --dry-run, or --yes for paid)
	$(UV) run python scripts/bench.py list

modal-deploy:       ## deploy the Modal runner app (needed for runner="modal")
	$(UV) run modal deploy workers/runners/modal_runner.py

smoke-rl:           ## one RL run on a Modal CPU container (~$0.001)
	$(UV) run python -m workers.runners.modal_runner smoke rl
