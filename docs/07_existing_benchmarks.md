# Existing benchmarks for ML-research agents, and which one we should use

## Rekursiv (rekursiv.ai): the closest competitor to our idea
- **What it is:** "AI Scientists" that form hypotheses, run experiments, review each other's work and update leaderboards. The founders are ex-Google Research, DeepMind and Luma (TensorFlow Probability, VideoPoet, PyMC, BlackJAX).
- **Self-reported claims:** new ML methods beating state of the art; a factory scheduler that won on 29 of 30 problems; a 15M-parameter RL agent beating Craftax's final boss in 44% of games vs 1 in 75 for "GPT-6 Astra"; state of the art on Karpathy's NanoChat in 3 days.
- **priml** (Apache-2.0): typed, configurable building blocks for A/B-testing ML research. The baselines are **nanochat** (a 300-second training budget, scored on validation bits-per-byte) and **sudoku**. It includes a FLOPs/roofline cost model. It ships no results tables.
- **trackinizer** (Apache-2.0): an "epistemological DB" of Issues, Artifacts and Beliefs linked by evidence edges, so every claim is traced to the experiments behind it. It ships no benchmarks.

**What this means for us:** Rekursiv is going after *"agents that do better research."* **Don't compete on that.** Our angle is the one they don't have: **the research agent's own LLM calls become cheap owned models that improve from corrections (Mubit)**, which means more experiments for the same budget. We could even use `priml`'s nanochat baseline as a target, since it's Apache-2.0 and adoptable within the event rules.

## Known benchmarks
| Benchmark | By | What it is | Fit for a 42-hour hackathon |
|---|---|---|---|
| **MLE-bench** | OpenAI | 75 Kaggle competitions, scored by medals. There is a "lite" split. | ❌ too heavy (GPUs, many hours per competition). Cite it, don't run it. |
| **RE-Bench** | METR | 7 open-ended ML research environments with a **fixed time budget**, scored against human experts. | ⚠️ Its time-budget methodology is exactly our claim. The tasks need GPUs. **Copy the method.** |
| **MLGym** | Meta | 13 AI-research tasks (vision, NLP, **RL**, game theory) in a Gym-style interface. | ✅ small tasks, RL included, standard harness. |
| **MLAgentBench** | Stanford | 13 ML experimentation tasks (improve a baseline `train.py`). | ✅ CPU-friendly tasks, well known. |
| **Karpathy-style speedruns / autoresearch** (nanoGPT speedrun, nanochat, fixed ~5-minute training) | community | Improve validation loss / bits-per-byte within a fixed training budget. | ✅ simple, viral, verifiable. Needs one GPU. |
| **DSBench / ScienceAgentBench / PaperBench / CORE-Bench** | various | data-science tasks, scientific coding, reproducing papers, reproducibility | Mostly off-target. Mention at most. |
| **Craftax** | Matthews et al. | JAX RL benchmark (Rekursiv uses it) | Possible RL target if we have a GPU. |

## Recommendation
Use a **standard benchmark harness plus RE-Bench's fixed-budget method**:
1. **Targets:** 3–4 **MLGym** or **MLAgentBench** tasks that train in under a minute on CPU (include one RL task), plus our **fraud** task for the bank story.
2. **Optional "wow" target:** a nanochat / speedrun-style task with a 5-minute training budget, if we get GPU credits (e.g. Modal).
3. **Arms:** Opus 5.5 agent vs Mubit-adapter agent vs a random-search floor, with the **same wall-clock and $ budget** and ≥3 seeds, scored on the benchmark's own held-out metric.
4. **Pitch line:** "On standard MLGym tasks, with the same 60-minute budget, our agent beats an Opus 5.5 agent, because Mubit makes each thinking step 20× faster and it keeps learning from researcher corrections."

Using a **published benchmark's tasks and scoring** is what makes the result believable. Judges can't dismiss it as a benchmark we built to win.

## Update: we have GPUs + CPUs. Upgraded plan
Run the parallel tracks on separate hardware:
| Hardware | Target | Why |
|---|---|---|
| **GPU (headline)** | **nanochat / nanoGPT-speedrun style**: a fixed 5-minute training budget, scored on validation bits-per-byte (priml's nanochat baseline is ready-made) | The most recognisable "AI lab" benchmark, the one Rekursiv brags about. One number, easy to verify. |
| **GPU (RL)** | **Craftax** (JAX, very fast on GPU) or an MLGym RL task | Covers "any model, including RL". Rekursiv's 44% figure gives us a reference point. |
| **CPU (bank story)** | Fraud PR-AUC (IEEE-CIS / Kaggle credit-card), plus 2–3 MLAgentBench tasks | The bank use case, quick iterations, many seeds. |

**Parallelism is the multiplier:** the Mubit agent's fast, cheap steps let it keep **N GPUs busy** at once, while an Opus agent bottlenecks on thinking time. Report **GPU utilisation %** next to the score, so the chart shows the Opus arm's GPUs sitting idle while it waits. This is Anil's "parallelism + agents wait less = full utilisation cycle" argument, measured directly.

**Fairness rule:** both arms get identical GPUs, the same wall-clock, the same $ cap, the same seeds and the same starting code. Only the LLM in the loop changes.

> Check the exact task lists and licences of MLGym / MLAgentBench on Friday night before committing. Pick the tasks whose training runs fastest.
