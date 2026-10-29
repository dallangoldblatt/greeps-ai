# Greeps AI

Learning policies for [Greeps](https://www.greenfoot.org/files/greeps/competition-flyer.pdf),
the Greenfoot programming competition: twenty identical aliens (greeps) must
collect tomatoes and bring them to their ship, each seeing only its own spot on
the map, with 8 bits of memory, 2 flags and paint as the only way to
coordinate.

This repo is meant to be read as well as run: each approach lives in its own
folder with its Java policy, training code and notes, so they can be compared
side by side, and new approaches can be added next to them.

## Layout

```
src/                  the Greenfoot scenario (open this in Greenfoot)
  Greep.java          the ACTIVE policy: a copy of policies/<name>/Greep.java
trained/              the best trained entry of each approach (+ checkpoints)
policies/             one folder per approach
  linear_cma/         linear policy, CMA-ES in the real scenario
  mlp_ppo/            neural network, MAPPO on the GPU simulator
  paint_comm/         same network, trained to communicate with paint
  README.md           policy interface, comparison, how to add an approach
greeps/               shared Python infrastructure
  java_sim.py         run the real scenario: headless runner, worker pool, GUI
  torch_sim.py        vectorized GPU port of the scenario (CUDA graphs)
  maps.py             scenario maps and custom .map files
  policies.py         which approach is active; embedding trained weights
  paint.py            paint ablations (scrambled / blind paint)
tools/                command-line tools
  use_policy.py       choose the active approach
  evaluate.py         score the active Greep in the real scenario
  paint_report.py     does a policy communicate with paint? (ablation, MI, heat maps)
  map_editor.py       browser map editor (+ map_editor.html)
  check_maps.py       validate maps (reachability, 40 px gaps, thin water) + previews
  generate_training_maps.py   randomized variants of every map family
  import_screenshots.py   screenshots of Greenfoot maps -> map files
  sim_gap.py          compare one policy in the Java and GPU simulators
  split_validation.py which map templates are held out for validation
headless/             HeadlessGreeps.java: the scenario without the GUI
lib/                  Greenfoot 3.7.0 and JavaFX 21 jars (Windows builds)
maps/                 custom maps (.map, with .png previews)
  training/           60 hand-designed maps, generators/, 1200 generated/ variants
  validation/         15 held-out templates + 300 generated/ variants
tests/                headless-vs-GUI and GPU-vs-Java parity tests
out/                  build output and training runs (not versioned)
```

**Maps.** Train on `maps/training`; compare methods and choose checkpoints
on `maps/validation`. The validation templates and their variants never appear in training
(`tools/split_validation.py`; `tools/generate_training_maps.py` keeps it so).

## Quick start

Run everything from the repository root. You need Python 3.12
(`pip install -r requirements.txt`; the GPU trainers need a CUDA build of
PyTorch) and a JDK 21 or newer: `JAVA_HOME`, a `javac` on `PATH`, or the
one bundled with VS Code's Java extension is found automatically
(`greeps/java_sim.py`). The jars in `lib/` are the Windows builds of JavaFX
21.0.3; on macOS or Linux replace `lib/javafx/*.jar` with your platform's
(controls, fxml, graphics, base, media, swing). All of this was developed
on Windows with an RTX 5070 Ti.

```powershell
python -m tools.use_policy                    # which approach src/Greep.java plays
python -m tools.use_policy linear_cma         # switch approach
python -m tools.evaluate                      # score it: 64 seeds x Maps 1-3
python -m tools.evaluate --gui                # ... and watch one run

# Train paint_comm with the latest recipe (6 hours; the other scripts in
# experiments/ are the comparisons in policies/paint_comm/README.md):
powershell -File policies/paint_comm/experiments/2026-10-23_long_all.ps1
python -m policies.paint_comm.export --run long_all

python -m tools.paint_report --maps maps/validation  # how the active policy uses paint
python -m tools.map_editor                    # draw maps, move ships and piles
python -m unittest discover -s tests -v       # GREEPS_SKIP_GUI=1 skips the GUI test
```

Trained weights are embedded in each approach's `Greep.java`, so whichever
one is in `src/` is a complete, self-contained competition entry.

## Results so far

The scenario's Maps 1-3 in the real (Java) scenario, 64 seeds, and the 315
validation maps on the GPU simulator, 8 episodes each; mean score per map.

| Approach | Map 1 | Map 2 | Map 3 | Maps 1-3 | Validation |
|---|---|---|---|---|---|
| hand-written start (linear_cma's initial weights) | 19.7 | 13.1 | 0.0 | 32.8 | |
| [linear_cma](policies/linear_cma/README.md) | 59.2 | 63.4 | 22.5 | **145.1** | 23.8 |
| [mlp_ppo](policies/mlp_ppo/README.md) (first run) | 44.9 | 58.6 | 12.1 | 115.6 | 39.0 |
| [paint_comm](policies/paint_comm/README.md) (`memory_st`) | 31.3 | 31.8 | 5.7 | 68.9 | **50.7** |

**Communication.** [paint_comm](policies/paint_comm/README.md) trains the
same network with incentives for using paint, on generated maps only (never
Maps 1-3), then a colour curriculum and trainable memory. Its greeps really
communicate: with paint scrambled between greeps they lose ~40% of their
score, mostly from *which* colour they read (carriers write orange along
delivery routes, searching greeps write red or purple). On the generated
validation maps it is the best approach by far; on Maps 1-3 it is the
worst, even on Map 3, which none of them trained on.

See [policies/README.md](policies/README.md) for the per-map comparison and
[trained/README.md](trained/README.md) for the best entry of each approach.

## Open problems (help welcome)

The full experiment log is in [policies/paint_comm/README.md](policies/paint_comm/README.md).
In short:

* **paint_comm's score plateaus at ~50 per validation map** whatever we
  change: shared team reward, longer memory sequences, an entropy floor on
  the paint colour with memory noise, 6-hour runs, or a 37x larger network
  (256 wide + 2 residual blocks). These change *how* paint is used, not the
  score. Training-map and validation scores match, so it is not
  overfitting either.
* **There is room to improve:** it delivers only 36% of the tomatoes on the
  validation maps (10% on the maze-like "built" maps, 22% on "terrain").
  How much is reachable in 120 s is unknown.
* **Generated maps rank the approaches differently from the scenario's
  Maps 1-3** (see the table above). Something about the generated maps
  differs in a way that matters.
* **Ideas not yet tried:** training from scratch with trainable memory and
  several seeds (every run so far descends from one early protocol);
  starting from a strong collector and adding communication; rewarding each
  paint drop by its counterfactual effect on the other greeps' returns; a
  full-observation teacher to measure the ceiling.
* **Constraint:** each greep keeps only the game's 8 memory bits + 2 flags
  between ticks; paint is the only shared channel. Training may use anything
  (privileged critic, auxiliary heads), as long as the deployed actor sees
  only what one greep may legally see.

## Two simulators

* **Java (ground truth).** `headless/HeadlessGreeps.java` runs the real
  scenario and Greenfoot classes without the GUI; with the same seed it
  replays the GUI run exactly (`tests/test_headless_parity.py`). About 8
  episodes per second on 14 cores.
* **GPU (training).** `greeps/torch_sim.py` re-implements the rules as
  tensors over thousands of maps, including Greenfoot's rotated-rectangle
  collision test; each tick is replayed as a CUDA graph. Not bit-identical
  (greeps act simultaneously), so `tests/test_torch_sim.py` checks that each
  approach scores the same in both simulators.

A Greenfoot quirk worth knowing: its collision tree iterates over HashMaps
keyed by identity hash codes, so when several greeps touch one loader, which
one gets the tomato depends on the JVM's history. Same-seed runs in fresh
JVMs match exactly; otherwise only the statistics match.

## Credits

The scenario in `src/` is based on Greenfoot's Greeps competition scenario
(`Earth.java` gained custom maps, seeds and traces; `Greep.java` is ours); `lib/` holds the Greenfoot 3.7.0 and OpenJFX 21 jars
(GPL v2 with the Classpath Exception).

