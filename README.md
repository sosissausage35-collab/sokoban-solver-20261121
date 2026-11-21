# Sokoban solver — per-room push-optimal A*

[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/sosissausage35-collab/sokoban-solver-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/sosissausage35-collab/sokoban-solver-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

Entry for make.fun `sokoban-solver-01-run-20261121` (as @sausage-sosis) — see the [make.fun bounty "Build the best Sokoban solver, 21 November 2026"](https://make.fun/bounty/sokoban-solver-01-run-20261121).

## Approach

Each level is a 1-wide hallway with 1–4 rooms hanging off it through a
single door (3–4 boxes and as many goals per room). A box pushed out of its
room can never return, so rooms are independent:

1. **Detect rooms** — find articulation cells whose removal separates a
   box/goal-balanced component from the player; the outermost cut wins, so
   the room keeps its whole door passage (a box may sit on the door cell
   itself, in which case the player starts on the passage cell beside it).
2. **Solve each room push-optimally** — A* over push-macro states (sorted
   box positions + canonical player position), with an optimal
   box→goal assignment heuristic over shortest-path distances, exact
   single-box liveness pruning, and 2×2 freeze pruning. The goal also
   requires the player to be able to walk back to the entry cell, so the
   next room can be reached (except for the last room).
3. **Stitch** — walk the player along the hallway between room entries
   (plain BFS, boxes as obstacles) and concatenate all moves.

Fully deterministic: fixed iteration orders, a sequence-numbered heap, and
budgets counted in node expansions, never wall-clock time. No `Date`, no
randomness, no network, no dependencies beyond Node built-ins.

If a room exhausts its optimal budget, weighted A* fallbacks return a valid
(possibly non-optimal) solution instead of nothing.

## Run

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`SOKOBAN_DEBUG=1` prints per-room diagnostics to stderr (never affects the
solution files).
