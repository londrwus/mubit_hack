# Night Shift London: rules, tracks, judging

**Fri 23 Oct, 7:30 PM hacking → Sun 25 Oct, 10:00 AM code freeze** (clocks go back at 2 AM, which gives us one extra hour). Expo judging at 10:30, finalist demos at 12:30, awards at 13:30.
Prizes: **$15k / $10k / $5k**, judged across all tracks, plus $35k in credits and Mubit credits.

## Hard rules (disqualifiers)
- **Every LLM call must route through Mubit.**
- All code must be written during the event. Open-source libraries are fine.
- At least one team member must attend the Sunday demo.

## Tracks
1. **Recurring tasks:** a repeated task (extraction, triage, support replies) where **corrections become model updates**.
2. **Own your intelligence layer:** move a closed-model workload to a task-adapted open model and **compare quality, cost and latency**.
3. **Fast refresh:** a changing workload where **model updates keep pace**.

## Judging (equal weight)
| Criterion | Verbatim | What it means for us |
|---|---|---|
| Use of Mubit | "Show a correction turning into a model update in your demo." | **Mandatory demo beat.** Show the before output, the human or verifier correction, the new adapter version, and the after output. Precompute it, because training is not instant. |
| Usefulness | "A real recurring task someone would want handled." | It must be high-volume and repetitive. One-off research does not count. |
| Technical execution | "It works, and it holds up under questions." | Show real numbers on a held-out set. Have honest answers about failure cases. |
| Demo | "Clear, live, and under the time limit." | Keep it to about 3 minutes, with one story, one chart, and one live call. |

## Practical constraints that will bite
- **25 kept examples minimum** per task. Version tests are graded **against the provider model**, so the adapter can at best *match* the frontier model on that task, unless corrections change the target.
- **3 adapter starts per org per day.** Training waits up to 20 minutes. There is a daily training limit and a pause between trainings. **Plan the demo so the "update" is already trained, and show the version history live.**
- A task is defined by an **identical prompt, tools and output schema**. Any prompt drift splits the task, so freeze prompts early.
