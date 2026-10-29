# Trained entries

The best trained policy of each approach as a finished competition entry: a
`Greep.java` with the weights embedded, needing nothing else. To play one,
copy it over `src/Greep.java` (or open `src/` in Greenfoot after copying);
the tools treat it like the approach's own file, so
`python -m tools.evaluate` works on it too.

| Entry | Network | Validation maps, per map (GPU sim) | Maps 1-3, total (Java, 64 seeds) | Checkpoint |
|---|---|---|---|---|
| [paint_comm](paint_comm/Greep.java) | 22-64-64-55, trainable memory | **50.7** (paint +20.6, colour +16.7, memory +4.9) | 68.9 | `paint_comm/checkpoints/memory_st_latest.pt` |
| [mlp_ppo](mlp_ppo/Greep.java) | 22-64-64-55 | 39.0 (paint +4.8, colour +0.2) | 115.6 | `mlp_ppo/run1_best.pt` |
| [linear_cma](linear_cma/Greep.java) | linear, 456 weights | 23.8 (paint +1.6; spits rarely) | **145.1** | `linear_cma/run1_best.weights` |

Validation: the 315 maps in `maps/validation`, 8 paired episodes each
(`python -m tools.paint_report --maps maps/validation --episodes 8`). Gains
compare real paint with paint scrambled between greeps, colours shuffled, or
each greep reading another greep's memory.

Caveats:

* The two columns disagree on the order. On generated maps paint_comm is
  best by far; on Maps 1-3 linear_cma and mlp_ppo are ahead. Both of those
  trained on Maps 1-2, while paint_comm never saw them, but they are also
  ahead on Map 3, which none of the three trained on (22.5 and 12.1 vs 5.7).
  Why the generated maps rank the approaches differently is an open
  question.
* paint_comm's entry is `memory_st` (trained with `--trainable-memory`), not
  the `colour_stage2` policy embedded in `policies/paint_comm/Greep.java`
  and described in its README: it scores higher on validation (50.7 vs
  47.9).

## paint_comm checkpoints

Training checkpoints to continue from, e.g.
`python -m policies.paint_comm.train --init-from <file> ...`:

| File | What it is |
|---|---|
| `colour_stage2_best.pt` | end of the colour curriculum (actor only); embedded in `policies/paint_comm/Greep.java` |
| `memory_st_latest.pt` | colour_stage2 + 1217 updates of trainable memory; this entry's weights |
| `night2_all_latest.pt` | memory_st + 100 min of the combined recipe (team share, 64-tick BPTT, colour-entropy floor, memory noise); start of the size runs |
| `h256b2_latest.pt` | 256 wide + 2 residual blocks (350k parameters), grown from night2_all, 1425 updates (`size3_h256_b1`). Validation 50.5 per map with the largest gains from paint (+19.2), colour (+15.0) and memory (+8.3). Too large to embed in Greep.java; needs loading from a file or an inference server |
