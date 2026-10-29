# paint_comm: greeps that learn to communicate with paint

```powershell
python -m policies.paint_comm.train --run main --minutes 120 --custom-maps maps/training `
    --select-maps "maps/training/*.map" --custom-weight 0.7 --real-weight 0 --random-maps 128 `
    --anneal-updates 2000 --game-anneal-updates 2000
python -m tools.paint_report --policy paint_comm --checkpoint out/paint/main/best_comm.pt --maps maps/validation
```

Then the colour curriculum (see [below](#colour-curriculum-stage-1--stage-2)),
whose result is embedded in `Greep.java`:

```powershell
# Stage 1: communication maps only, colour-specific incentives (45 min)
python -m policies.paint_comm.train --run colour_stage1 --minutes 45 --init-from out/paint/main/latest.pt `
    --custom-maps "maps/training/**/signal_*.map" --select-maps "maps/training/signal_*.map" `
    --custom-weight 1.0 --real-weight 0 --random-maps 0 `
    --colour-incentives --colour-diversity-coef 0.01 --incentive-floor 0.2 `
    --anneal-updates 1500 --game-seconds 240 --game-anneal-updates 1000
# Stage 2: the full map pool (90 min)
python -m policies.paint_comm.train --run colour_stage2 --minutes 90 --init-from out/paint/colour_stage1/latest.pt `
    --custom-maps maps/training --select-maps "maps/training/*.map" --custom-weight 0.7 --real-weight 0 `
    --random-maps 128 --colour-incentives --colour-diversity-coef 0.01 `
    --incentive-floor 0.2 --anneal-updates 1500 --game-seconds 240 --game-anneal-updates 1000
python -m tools.paint_report --policy paint_comm --checkpoint out/paint/colour_stage2/best.pt --maps maps/validation
python -m policies.paint_comm.export --checkpoint out/paint/colour_stage2/best.pt
```

Same network as [mlp_ppo](../mlp_ppo/README.md) (22 legal observations,
including the three paint colours under the greep; sampled heads). What is
new is how it is trained.

## Why greeps did not use paint before

* **No credit for spitting.** A drop helps another greep, maybe hundreds of
  ticks later. Per-greep rewards and short credit horizons never connect the
  two.
* **Chicken and egg.** Nobody reads paint because it means nothing, and
  nobody writes meaning because nobody reads it.
* **Physics.** A walking greep can drop one 21 px blob every 10-19 moves, and
  senses only the colours directly under it: no direction, no trails for free.

mlp_ppo spat constantly (73% of decisions), so paint covered the map and only
its *density* said anything (+4.8 per validation map from paint, +0.2 from
colour).

## Training (`train.py`)

On top of MAPPO:

* **Influence reward.** When a greep stands on paint spat by another greep,
  its movement decision (mode, turn, move) is compared with what it would do
  with a *scrambled* paint reading (another greep's, so the same statistics
  but no information; `greeps/paint.py`). The KL divergence, capped at 0.3 per
  reader and tick, is credited to the drop's author, written back to the tick
  the drop was spat (`GreepsSim.paint_step`) so the spitting action itself is
  reinforced.
* **Positive listening:** a loss term rewarding readers whose movement depends
  on others' paint (capped). **Positive signalling:** a loss term rewarding
  spit choices that depend on the writer's situation (batch mutual
  information).
* **Spit cost** of 0.005 per drop.
* The three incentives fade to zero over 2000 updates, so only paint use that
  pays off in tomatoes survives.
* **Game-length curriculum:** training games start at 360 s and shrink to
  the real 120 s, so hard maps give enough successes and enough readers per
  drop to learn from. The actor never sees the clock.
* **Maps:** 1260 training maps (`maps/training`: 60 hand-designed + 1200
  variants) and 128 random maps; never the scenario's Maps 1-3
  (`--real-weight 0`).

Getting the balance right took two failed starts, both worth knowing:

| Run | Influence | Spit cost | Outcome |
|---|---|---|---|
| `out/paint/main_spam` | uncapped, 0.02 | 0.01 | **Influence gaming**: greeps spammed paint that jolted readers (99.5% of greeps on paint, influence 3.6 nats per read); the task collapsed (~0.3 tomatoes per map). |
| (stopped) | capped 0.3, 0.005 | 0.02 | **Extinction**: spitting fell to 0.07 per 100 ticks within 20 updates; nothing left to learn from. |
| `out/paint/main` | capped 0.3, 0.005 | 0.005 | Stable: ~4 drops per 100 ticks, paint used. |

## Results (`out/paint/main`, 120 minutes, ~3000 updates)

Paint ablation on the 315 held-out validation maps (GPU simulator, 8 paired
episodes per map), mean per map. *Scrambled* gives every greep a random other
greep's paint reading: same amount of paint, no information. *Colours
shuffled* gives each greep on paint the colours under a random other greep
on paint (presence of paint and the colour mix kept,
`paint_colours_shuffled`).

| Checkpoint | With paint | Scrambled | Gain from paint | Gain from colour | Paint hidden |
|---|---|---|---|---|---|
| best.pt | **47.1** | 29.6 | +17.5 ± 0.2 | +12.3 | 6.9 |
| best_comm.pt | 45.8 | 26.1 | **+19.7 ± 0.2** | **+16.4** | 6.2 |
| mlp_ppo, for comparison | 39.0 | 34.1 | +4.8 ± 0.1 | +0.2 | 3.9 |

Most of the information is in *which* colour, with red a rare, strong
marker. With paint hidden entirely the score collapses: like mlp_ppo, the
policy never saw "no paint" in training, which is why the scrambled
comparison is the meaningful one. `best.pt` is the best training-map score
(update 1975, incentives fully off); `best_comm.pt` the largest
training-map information gain (update 1800). (Since replaced in `Greep.java`
by the colour curriculum below.)

## What the greeps do with paint (main run)

* **Mostly one colour.** ~90% orange, ~8% red, ~2% purple. Colour carries a
  little information about the writer (0.05-0.06 bits about being near the
  ship or carrying; mlp_ppo < 0.03), but the protocol is mainly *where* paint
  is, not which colour.
* **Carriers write more:** 5.8 drops per 100 ticks when carrying vs 3.8
  empty-handed, so trails thicken along routes that deliver tomatoes.
* **Paint marks routes and shorelines.** The heat maps
  (`out/paint_report/paint_comm/*/heatmaps.png`) show sparse lines along
  delivery routes and concentrated along water edges and around lake tips,
  where greeps have to turn, instead of mlp_ppo's map-wide cloud.
* **Readers steer by it.** On another greep's paint, 35% of decisions differ
  from those with a scrambled reading, mostly the turn (21%) and heading mode
  (14%).

## Colour curriculum (stage 1 → stage 2)

To get the greeps to use all three colours, training continues from
`out/paint/main/latest.pt` with `--colour-incentives`:

* The influence reward and listening term compare a reader's decision with
  *colours shuffled* (`shuffle_colours`) instead of scrambled paint, so only
  the change caused by which colour is credited.
* Signalling rewards mutual information between colour choice and the
  writer's situation, plus a colour-diversity bonus (`--colour-diversity-coef`,
  entropy of the batch's colour mix).
* The incentives anneal to 20% (`--incentive-floor 0.2`), not to zero.

**Stage 1** (45 min) trains only on the 315 communication maps
(`maps/training/signal_*` and their variants: forks, depleting piles, mirror
twins, ring roads, ...), where reading another greep's colour pays off most.
**Stage 2** (90 min, ~1500 updates) resumes on the full pool (1575 training
maps + 128 random maps). Validation maps, GPU simulator, 8 paired episodes
per map, mean per map:

| Checkpoint | With paint | Scrambled | Colours shuffled | Gain from paint | Gain from colour |
|---|---|---|---|---|---|
| main/best.pt | 47.1 | 29.6 | 34.9 | +17.5 | +12.3 ± 0.2 |
| main/best_comm.pt | 45.8 | 26.1 | 29.4 | +19.7 | +16.4 ± 0.2 |
| **colour_stage2/best.pt (embedded)** | **47.9** | 23.0 | 22.8 | +24.9 | **+25.1 ± 0.3** |
| colour_stage2/best_comm.pt | 46.5 | 13.0 | 13.1 | +33.5 | +33.3 ± 0.2 |

So the score held while half or more of it came to depend on reading
colours.

**What the colours mean** (`tools/paint_report`, colour mix of drops by the
writer's situation):

| Writer | red | orange | purple | Drops |
|---|---|---|---|---|
| carrying a tomato | 0% | 99% | 1% | 22% of drops |
| empty-handed | 25% | 36% | 39% | 78% |
| near a pile (< 60 px) | 5% | 87% | 8% | |

* **Orange = "tomatoes came this way".** Carriers write orange almost
  exclusively, so orange lines mark working routes between piles and the
  ship (visible as thin straight orange streaks in
  `out/paint_report/paint_comm/colour_stage2/best/heatmaps.png`).
* **Red and purple = "searched here, nothing carried".** Only empty-handed
  greeps write them, as a diffuse wash over explored ground; heading also
  shifts the red/purple balance a little (0.04 bits).
* Colour now carries 0.25 bits about the writer (main: 0.05-0.06), and on
  another greep's paint, 69% of decisions change when only the colours are
  shuffled (main: 18-20%). The colour mix went from 90/8/2% to 20/50/30%.

Caveat: with scrambled or shuffled colours the score halves (best_comm.pt:
drops to a quarter), so the policy now depends heavily on the protocol; it would be fragile next
to greeps that paint differently.

## Trainable memory (`--trainable-memory`)

Until here the 8 memory bits and 2 flags were sampled actions, trained only
by reward many ticks later. The stage 2 policy's bits were close to coin
flips (certainty |2p - 1| ≈ 0.13-0.24): the greeps barely used memory.

With `--trainable-memory` the bits are a recurrent state trained by gradient
(straight-through bits, PPO replayed in 32-tick sequences, soft memory
hardening to real bits over 300 updates, a fading warm-up asking memory to
predict pile and ship facts; see `train.py`). Evaluation and `Greep.java`
still sample hard bits, unchanged.

Both runs continued from colour_stage2 for 120 minutes with the incentives at
their 20% floor and real 120 s games; only memory training differs.
Validation maps, `latest.pt`, GPU simulator, 8 paired episodes per map,
mean per map:

| | Control (`size_h64`) | Trainable memory (`memory_st`) |
|---|---|---|
| Updates | 1426 | 1217 |
| Score | 49.6 | **50.7** |
| Gain from paint | +14.2 ± 0.3 | **+20.6 ± 0.3** |
| Gain from colour | +9.7 | **+16.7** |
| Gain from memory (each greep reads another's memory) | +0.4 ± 0.4 | **+4.9 ± 0.3** |
| Memory bit certainty | 0.15-0.28 | **0.40-0.77** |
| I(memory byte; colour underfoot) | 0.04 bits | **0.38 bits** |
| I(memory byte; carrying) | 0.19 bits | **0.34 bits** |
| Readers whose memory write is changed by paint | 32% | **50%** |

* Memory now stores what the greep has been reading: colour, being on
  paint, carrying, near a pile. Remembering the last colour lets a greep
  notice when the colour under it changes, which a single reading cannot.
  It does not store pile direction (≈ 0.01 bits), despite the warm-up.
* With memory, paint stayed useful: the control drifted towards ignoring
  paint (gain from colour +25.1 at stage 2 → +9.7), the memory run kept
  +16.7 at a similar score.
* The score barely moved: 50.7 against colour_stage2/best.pt's 47.9 and the
  control's 49.6. In Java on Maps 1-3 it scores 68.9 against
  colour_stage2/best.pt's 75.3, which stays embedded in this folder's
  `Greep.java` ([trained/](../../trained/README.md) uses memory_st).
* Chasing an apparent Java gap for this policy (`tools/sim_gap.py`) found a
  real simulator bug: the GPU sim's paint contact test used real-valued
  half-sizes and counted 5% more greep-drop contacts than Greenfoot, all at
  blob edges. It now reproduces Actor.intersects exactly (Java agrees on all
  but ~1 in 300,000 pairs) at ~0.4 s more per rollout. The rest of the
  apparent gap was noise from small Java samples (8-32 episodes; the score
  varies a lot between episodes). With the fix the two simulators agree to
  within ~2%; the old test had flattered paint-heavy policies by 2-4%.

Probe: `tools/paint_report` prints the memory ablation; per-bit statistics
came from a scratch probe on the 15 validation templates (mutual
information of each bit with the situation).

## Credit, longer memory and exploration (`night2_*`, validation maps)

From here on, checkpoints are chosen and methods compared on
`maps/validation`: 15 templates (3 per family) and their 300 variants, moved
out of training by `tools/split_validation.py`.

Five 100-minute arms (~950 updates each) continued from `memory_st/latest.pt`
with the same recipe, each changing one thing
(`experiments/2026-10-22_recipe_comparison.ps1`; the other experiments
below have their scripts in `experiments/` too):

* `--team-share 0.5`: half of each delivery's reward is split over all greeps.
* `--bptt 64 --minibatches 4`: 64-tick memory sequences (16 instead of 32
  gradient steps per update, to keep the speed).
* `--colour-entropy-target 0.2 --memory-noise 0.02`: an adaptive bonus keeps
  the entropy of the spit colour above 0.2 nats, and 2% of stored bits flip
  (training only).
* all three together.

`tools/paint_report` on `best.pt`, 315 validation maps, 8 paired episodes,
per map (gains ± ~0.2-0.3):

| Arm | Score | Gain from paint | from colour | Gain from memory | Colours (r/o/p) | I(colour; carrying) |
|---|---|---|---|---|---|---|
| start (`memory_st`, first eval) | ~50.2 | ~20.5 | ~17 | | | |
| base | 51.6 | +18.5 | +12.7 | +7.9 | 6/64/30 | 0.18 |
| team share | 51.0 | +16.3 | +10.7 | +5.8 | 6/62/32 | 0.20 |
| bptt 64 | 50.8 | +19.0 | +14.1 | +7.8 | 7/58/35 | 0.21 |
| explore | 52.1 | +15.4 | +6.7 | +5.9 | 11/45/44 | 0.34 |
| all | 51.4 | +18.1 | +12.9 | +8.0 | 8/49/43 | 0.32 |

* No arm clearly improves the score. Single selection evals vary by ±0.6,
  and `best.pt` is the best of ~37 evals on these same maps, which flatters
  every arm by about a point. Averaged over each run's last quarter, the
  scores are 50.3 / 50.2 / 49.1 / 50.8 / 51.0.
* Without the entropy floor (base, team, bptt 64), paint slowly loses value:
  colour gain over the run fell 15 → 13, 14 → 10.5 and 18 → 12. With it
  (explore, all), colour gain grows during the run (7.5 → 12.7 and 7.9 → 10.3,
  from a dip at the start), colours are used more evenly, and colour carries
  more about the writer (0.32-0.34 bits about carrying vs 0.18-0.21).
* Team share alone hurt both paint and memory use.
* `all` is the only arm whose score rose steadily over the run (quarters
  50.4 → 50.6 → 50.9 → 51.0) while keeping paint and memory gains at base
  level.

Running `all` on for 6 more hours (`long_all`, 3071 updates) did not help.
Selection scores drifted from 50.5 to 49.9 over the run.

| Checkpoint | Score | Gain from paint | from colour | Gain from memory | Blind |
|---|---|---|---|---|---|
| `best.pt` (best of 122 evals) | 51.4 | +15.1 | +8.5 | +7.9 | 14.8 |
| `latest.pt` | 49.4 | +13.8 | +3.1 | +5.8 | 23.7 |

The policy moved towards behaviour that works without paint (the blind score
rose by 60%) and almost stopped using colour. This 64-wide actor seems to
have plateaued at ~50 per validation map; the next test is network size.

Network size (`size2_*`, same recipe and start, 3 hours each):

| Run | Updates | `best.pt` score | Gain from paint | from colour | Gain from memory |
|---|---|---|---|---|---|
| 64 wide (`size2_h64`) | 1556 | 51.3 | +16.0 | +8.6 | +6.1 |
| 256 wide, 2 blocks (`size2_h256b2_mb16`) | 116 usable | 51.1 | +18.0 | +12.4 | +9.5 |

* The big actor (350k parameters) needed 16 minibatches to fit in GPU memory
  (4 overflowed into system RAM) and runs ~4x slower per update.
* It diverged at update 117: policy entropy halved over updates 102-112, then
  the losses went NaN. Its best checkpoint (from update ~50) already used
  paint and memory more than the 64-wide one, at the same score.
* The 64-wide control again drifted away from colour (`latest.pt`: 49.0,
  colour gain +3.0).
* `train.py` now stops on a non-finite loss instead of training on NaN
  weights.
* Retry at lr 1e-4 with the colour coefficient capped at 0.05
  (`size3_h256_a1`, 489 updates): no NaN, but the policy still collapsed
  (entropy 1.13 → 0.46, selection score 51 → ~45; `latest.pt` 43.9 with no
  gain from colour or memory). Its best checkpoint is the first eval
  (update 25). With 16 minibatches × 4 epochs it takes 64 gradient steps per
  update against the control's 16; `size3_h256_b1` (1 epoch, the same 16
  steps, ~5 s per update) and `size3_h256_b2` (4 epochs, lr 5e-5) follow.
* `size3_h256_b1` (1425 updates) trained stably and kept using paint: its
  `latest.pt` scores 50.5 with gains from paint +19.2, colour +15.0 and
  memory +8.3, where the 64-wide control ended at 49.0 / +13.7 / +3.0 /
  +4.1. The score itself did not rise (quarter averages 50.9, 50.6, 50.9,
  50.6), so capacity is not what limits the score.
* `size3_h256_b2` (4 epochs at lr 5e-5, 482 updates) also stayed finite and
  flat (quarters 50.7, 50.8, 50.2, 50.7; `latest.pt` 50.1, paint +17.5,
  colour +12.6, memory +4.9), but its policy entropy kept falling
  (1.05 → 0.82), so 1 epoch is the better setting for the big actor.

The first attempt (`night_*`) is not comparable: a GPU memory leak in
checkpoint selection (a new CUDA stream per graph capture, so cuBLAS
workspaces piled up) slowed every run 20-100x part-way, leaving 61-236
updates. Fixed in `torch_sim.capture_graph`.

## Limits and next steps

* On the scenario's Maps 1-3 (Java, 64 seeds) the embedded policy scores
  75.3 in total, far below linear_cma (144.5) and mlp_ppo (113.8). Those
  trained on Maps 1-2; on Map 3, which none of them trained on, it is
  between the two (15.5 vs 22.6 and 11.2).
* The protocol is two-valued (carrying vs searching), with red and purple
  nearly interchangeable. Directional meaning (e.g. "ship is that way") is
  the obvious next step; the speaker only weakly encodes heading and
  ship/pile direction (<= 0.04 bits).
* Stage 2 was limited by time (~1500 updates); scores were still noisy, not
  clearly converged.
