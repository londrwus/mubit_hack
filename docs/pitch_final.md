# Final pitch to the judges (2 minutes max, last 20 s = live dashboard test)

Written Sun 25 Oct 2026 for the final frontend (Landing v3: How it works → Cost → Time → Model score → Benchmarks; dashboard `/app`). It follows the landing order, so the screen and the words move together. Numbers come only from `results_log.md` (R-ids in brackets, not spoken) and follow `pitch.md` §0, §3 and §7. **Budget: about 1:15 of speech (about 180 words), then the 20 s live test; about 1:35 in total.**

**Style: Jordan Belfort's Straight Line, toned down.** It uses his tools without the hype:
- **Certainty from the first sentence:** short declarative lines, no "we think" or "kind of".
- **Pain first, then the fix:** open with a question that makes them feel the waste.
- **The three tens, in order:** the problem is real, the product fixes it (with numbers), and we can be trusted (the logged limits).
- **"Reasonable man" tone on the limit:** slow down, say the loss plainly. Admitting it is what makes the rest believable.
- **Proof over promises:** "Don't take my word for it. Watch."

Not used, on purpose: shouting, fake urgency or scarcity, "ladies and gentlemen", superlatives we can't back, and more than one rhetorical question in a row.

The tabular result (R82) is left out for time, so its failed first try (R75) doesn't need saying. The real-GPU limit (R85) stays: it must appear wherever results are shown.

## Before going on stage
- Backend running with the logged races (`docs/dashboard_api.md` §1) and `npm run dev` in `frontend/`.
- Tab 1: `http://localhost:5173/`, scrolled to the top. Tab 2: `http://localhost:5173/app`. Click **Test key** once (an empty field checks the server's key) so the first live call is warm, then scroll to **"See it work in 10 seconds"**.
- One live run costs about $0.07 of Opus through Mubit (R94). Rehearse with **"Rehearse free (mock AI)"**, which costs nothing, and save the real button for the stage.
- Fallbacks, in order: (1) **Rehearse free (mock AI)**, said as "a rehearsal with a simulated AI"; (2) `/app/compare` → Game-playing sweep → Replay at 90×, said as "a replay of a logged race".

## The script

| Time | Screen | Say | Delivery |
|---|---|---|---|
| 0:00–0:10 | Landing hero | 
"Your lab lets an AI plan its training. What are you paying for? 
Mostly, the AI thinking. And while it thinks, your machines wait."

| 0:10–0:22 | **I · How it works** | 
"Busy Machines is **compute intelligence for training**. 
It calls the AI only when it's worth it, 
keeps every machine working in between, and 
checks every win before it ships. All through Mubit."

| 0:22–0:38 | **II · Cost** | 
"Same Opus 5.5, same machines: 
**the same model, for a quarter of the AI bill.** 
Eighty-three cents, not three-thirty. 
And Mubit is turning those calls into small models we own. 
They're not serving yet; when they do, it gets cheaper still."

| 0:38–0:48 | **III · Time** | 
"Their machines waited **sixty-nine percent** of the time. 
Ours? **Two.** Same fifteen minutes, 
**three times the experiments.**"

| 0:48–0:58 | **IV · Model score** | 
"And **zero** of eight hundred forty-seven planted fake wins got through. 
I've fine-tuned on H100 GPU the model, Sapling, based on the Mubit model from hugging face, 
my model caught **fifty-seven** real fakes; the published model caught none."

| 0:58–1:10 | **V · Benchmarks** | 
Here's benchmarks. On long real-GPU runs we were cheaper on AI, 
time and better on efficency and costs. It's all in our public log. 
Short, fast sweeps are our lane."

| 1:10–1:14 | Switch to the `/app` tab | 
"Don't take my word for it. Watch."

"Same Opus, sixteen machines each. Left: waiting. Right: already working." 

| **1:25–1:35** | Result box | (read **off the screen**, not from memory) "**Nine seconds**, versus **a tenth of one**. **[N] experiments to zero.** That's live. Busy Machines." | Stop there. |

About 1:35 in total, leaving 25 s of buffer inside the 2-minute limit (for a slow start, applause, or a slow first AI call).

## What the live numbers mean (say it right)
- R94, one real run: the first experiment started after 9.33 s vs 0.06 s; machines busy 3% vs 99%; experiments 0 vs 46. The live run will differ a little, so always read the screen.
- The **0.06 s is not a faster AI answer**. Busy Machines' own Opus call takes about 6 s. Its machines start on runs from the no-AI tuner without waiting for that call. Say "already working" or "doesn't wait", never "answers faster".
- **Don't mention the AI bill in the live test.** In 10 s, Busy Machines' bill was slightly higher (one extra checker call, +$0.007, R94). The bill claim comes from the 15-minute races (R78).
- The panel says "demo run, not a benchmark result", and that's correct.

## Sources for every spoken number
| Line | Source |
|---|---|
| Same model (a tie), $0.83 vs $3.30, three runs | R78 (game-playing) |
| "Our engine alone", Mubit models not serving yet | R70, R79, R80 |
| 69% vs 2% waiting, 30 machines | R78 (68.8% vs 1.5%, 15 s runs) |
| Three times the experiments in 15 min | R78 (1,587 vs 500) |
| 0 of 847 planted shortcuts | R78 |
| 57 of 109 real fakes caught vs none (published model) | R81; public: R92 |
| Real GPUs: 4.5× cheaper on AI, 5% worse | R85 |
| Live: about 9 s vs under 0.1 s; experiments | read off the screen (R94 for reference) |

## If you're over time
There is 25 s of buffer. If still over, cut "It's all in our public log." first. **Never cut** the real-GPU limit (0:58–1:10), "They're not serving yet", or "a quarter of the AI bill" without "the same model".

## Q&A
Use `pitch.md` §6. Keep the same tone: short, certain, and admit a limit plainly when there is one. Most likely questions: "Isn't that what Mubit does?", "Is it just Optuna doing the work?", "Did it work on other tasks?" (answer: tabular R82, a quarter of the AI bill, Opus slightly ahead in every run, chosen after R75 failed), and "Show me a correction turning into a model update". Answer the last one honestly: each correction is stored against its Mubit request ID; the models are trained but none has passed its exam yet (R79, R80).
