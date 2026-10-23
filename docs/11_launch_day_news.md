# ⚠️ Launch-day news: Mubit announced 3 things TODAY (23 Oct 2026)
Source: Anil's launch post (x.com/notanilp/status/2113651825752629256), summarised by Digg. Check the originals at kickoff.

## 1. Mubit platform (public launch, launch week 26–30 Oct)
- "Turns your agents' work into models you own." Dedicated models for **recurring tasks**, with frontier models kept for everything else.
- **Switches to the dedicated model only when it performs better.** That is routing, so adapters *are* used at inference. This answers our earlier open question in principle; confirm the details at office hours.
- Third parties report a one-line base-URL swap and **traces refreshing every few minutes**.
- **Critics' objection:** "won't agents just repeat existing tasks?" Our pitch answers this directly: research is the case where the *recurring steps* (propose, diagnose, judge) power *non-repeating* discovery.

## 2. Banyan Sapling: their first open model (Apache 2.0, Hugging Face `MubitLabs/banyan-sapling`)
- A research preview that **checks whether a claim is supported by the supplied policy and evidence**. It flags conflicts, missing info and inconsistent context.
- Model card: 94.22% raw accuracy on 1,731 cases. Calibration is weak on ContractNLI (0.82%). They state it has **no proven real-world accuracy**.
- **For us:** this is a *verifier*, which matches Anil's "keep the verifier separate from the context" thesis. **Use Sapling as the independent checker in our `judge_improvement` step:** "claim: this run improved val_bpb by 4%; evidence: eval config, data split hashes, logs." It catches leakage or unsupported gains, and its flags become the **corrections** that train the adapter.
  - This uses their brand-new model in a domain they didn't design it for (ML research claims, not policy or contracts). Judges will notice.
  - Rule check: Sapling runs locally (open weights), so it's not an LLM API call. To stay safe, **ask whether it must go through Mubit**, or serve it on Modal and keep every LLM call routed through Mubit.

## 3. latte/bench: "a benchmark for whether agents actually learn" (mubit.ai/bench)
- Agents plan simulated latte pours, get feedback and retry, with and without a retained note.
- They state it measures **client-side notes, "not model-weight learning"**, and **does not test held-out transfer**.
- **For us, this is our opening:** latte/bench explicitly leaves out *weight learning on real tasks*. Our benchmark measures exactly that. Mubit adapters learn in the weights from corrections, and we test on **real ML/DL/RL tasks with held-out metrics**, costed in GPU $.
  - Pitch line: **"latte/bench asks whether agents learn. We measure what that learning is worth: dollars and hours to a real research result."**
  - Name idea: **"lab/bench"** or **"night/bench"**, matching their naming style.

## Updated pitch stack
1. **Mubit platform:** the agent's recurring calls become owned adapters, so it is faster and cheaper.
2. **Banyan Sapling:** an independent verifier of research claims, whose flags are corrections.
3. **Our benchmark (night/bench):** $-to-target and GPU idle %, Opus 5.5 agent vs Mubit agent, on real nanochat / RL / fraud tasks.

We would be the first team to use all three of their launches together, in a market they haven't addressed, on their launch weekend.
