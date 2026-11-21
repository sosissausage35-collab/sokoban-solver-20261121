// Sokoban solver: per-room push-optimal A* with room decomposition.
//
// Idea (from the level structure, not from any other solver): every level is
// a 1-wide hallway with 1-4 rooms hanging off it through a single door, each
// room holding 3-4 boxes and as many goals. A box pushed out of its room can
// never come back, so each room is independent: detect the rooms by looking
// for articulation cells whose removal separates a box/goal-balanced
// component from the player, solve each room push-optimally on its own with
// A* over pushes, then stitch the room solutions together with plain walks
// along the hallway.
//
// Per room, A* runs over push-macro states: the state is the sorted box
// positions plus the player's canonical position (the smallest floor cell
// reachable from the player given the boxes), which merges equivalent
// states. The heuristic is the optimal assignment of boxes to goals over
// shortest-path distances ignoring other boxes (admissible and consistent),
// dead cells are pruned with an exact single-box liveness analysis, and
// 2x2 frozen blocks are pruned. Everything is deterministic: fixed
// iteration orders, a sequence-numbered heap, and budgets counted in node
// expansions, never in wall-clock time. No Date, no randomness, no network.
//
// Local run:
//   MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
// Set SOKOBAN_DEBUG=1 for per-room diagnostics on stderr (never affects
// the solution files).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DEBUG = process.env.SOKOBAN_DEBUG === '1';

const log = (...args: unknown[]): void => {
  if (DEBUG) console.error(...args);
};

type Dir = { dx: number; dy: number; ch: string };
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];
const OPP = [1, 0, 3, 2];

// ---------------------------------------------------------------- level ---

interface Level {
  w: number;
  h: number;
  floor: Uint8Array; // 1 when not a wall
  goal: Uint8Array;
  boxes: number[]; // sorted global cell indices
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const floor = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      floor[i] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, floor, goal, boxes, player };
}

function globalNeighbours(idx: number, w: number, h: number): number[] {
  const x = idx % w;
  const y = (idx / w) | 0;
  const out: number[] = [];
  for (let d = 0; d < 4; d++) {
    const nx = x + DIRS[d]!.dx;
    const ny = y + DIRS[d]!.dy;
    out.push(nx < 0 || ny < 0 || nx >= w || ny >= h ? -1 : ny * w + nx);
  }
  return out;
}

/** Connected components of the floor graph with one cell removed. */
function componentsMinus(level: Level, removed: number): number[][] {
  const { w, h, floor } = level;
  const seen = new Uint8Array(w * h);
  if (removed >= 0) seen[removed] = 1;
  const comps: number[][] = [];
  const stack: number[] = [];
  for (let s = 0; s < w * h; s++) {
    if (!floor[s] || seen[s]) continue;
    const comp: number[] = [];
    stack.length = 0;
    stack.push(s);
    seen[s] = 1;
    while (stack.length > 0) {
      const u = stack.pop()!;
      comp.push(u);
      const ns = globalNeighbours(u, w, h);
      for (let d = 0; d < 4; d++) {
        const v = ns[d]!;
        if (v >= 0 && floor[v] && !seen[v]) {
          seen[v] = 1;
          stack.push(v);
        }
      }
    }
    comps.push(comp);
  }
  return comps;
}

// ------------------------------------------------------- room detection ---

interface Candidate {
  door: number; // articulation cell separating the room from the player
  cells: number[]; // the separated component (without the door), sorted
  roomCells: number[]; // component + door, sorted
  boxes: number[]; // global box cells inside roomCells, sorted
  goals: number[]; // global goal cells inside roomCells, sorted
}

function collectCandidates(level: Level, minB: number, maxB: number): Candidate[] {
  const { w, h, floor, boxes, goal, player } = level;
  const boxSet = new Set(boxes);
  const cands: Candidate[] = [];
  const cellList: number[] = [];
  for (let i = 0; i < w * h; i++) if (floor[i]) cellList.push(i);
  for (const a of cellList) {
    for (const comp of componentsMinus(level, a)) {
      if (comp.indexOf(player) >= 0) continue;
      let nb = boxSet.has(a) ? 1 : 0;
      let ng = goal[a] ? 1 : 0;
      for (const c of comp) {
        if (boxSet.has(c)) nb++;
        if (goal[c]) ng++;
      }
      if (nb !== ng || nb < minB || nb > maxB) continue;
      const cells = [...comp].sort((x, y) => x - y);
      const roomCells = [...cells, a].sort((x, y) => x - y);
      const rboxes: number[] = [];
      const rgoals: number[] = [];
      for (const c of roomCells) {
        if (boxSet.has(c)) rboxes.push(c);
        if (goal[c]) rgoals.push(c);
      }
      cands.push({ door: a, cells, roomCells, boxes: rboxes, goals: rgoals });
    }
  }
  // Prefer the outermost cut (largest room): it contains the whole room
  // plus its door passage, so the room solver never lacks a manoeuvring
  // cell. Ties break deterministically on the door cell.
  cands.sort(
    (x, y) =>
      y.roomCells.length - x.roomCells.length ||
      x.door - y.door ||
      x.cells[0]! - y.cells[0]!,
  );
  return cands;
}

/** Pick candidates that partition all boxes into disjoint rooms. */
function partitionRooms(level: Level, cands: Candidate[]): Candidate[] | null {
  const chosen: Candidate[] = [];
  const claimed = new Uint8Array(level.w * level.h); // claimed component cells
  const chosenDoors: number[] = [];
  const byBox = new Map<number, Candidate[]>();
  for (const c of cands) {
    for (const b of c.boxes) {
      const arr = byBox.get(b) ?? [];
      arr.push(c);
      byBox.set(b, arr);
    }
  }
  const dfs = (uncovered: number[]): boolean => {
    if (uncovered.length === 0) return true;
    const b = uncovered[0]!;
    const list = byBox.get(b) ?? [];
    for (const cand of list) {
      if (!cand.boxes.every((x) => uncovered.indexOf(x) >= 0)) continue;
      let ok = true;
      for (const c of cand.cells) {
        if (claimed[c]) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      if (claimed[cand.door]) continue; // this door lies inside another room
      for (const d of chosenDoors) {
        if (cand.cells.indexOf(d) >= 0) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      for (const c of cand.cells) claimed[c] = 1;
      chosenDoors.push(cand.door);
      chosen.push(cand);
      const next = uncovered.filter((x) => cand.boxes.indexOf(x) < 0);
      if (dfs(next)) return true;
      chosen.pop();
      chosenDoors.pop();
      for (const c of cand.cells) claimed[c] = 0;
    }
    return false;
  };
  return dfs([...level.boxes]) ? chosen : null;
}

// ----------------------------------------------------------------- room ---

interface Room {
  w: number;
  h: number;
  door: number; // global cell of the separating articulation
  entry: number; // global cell where the player starts this room
  cells: number[]; // global solver floor (room + door-passage neighbours), sorted
  local: Int32Array; // global cell -> local index, -1 when not in this room
  n: number;
  neigh: Int32Array; // n * 4: local neighbour per direction, -1 when none
  cellX: Int32Array; // local -> global x
  cellY: Int32Array; // local -> global y
  goals: number[]; // local goal cells, sorted
  goalSet: Uint8Array;
  goalCode: number;
  boxes0: number[]; // local starting box cells, sorted
  goalsGlobal: number[];
  boxesGlobal: number[];
  entryLocal: number;
  dist: Int32Array; // n * n shortest-path distances ignoring boxes, -1 = unreachable
  live: Uint8Array; // exact single-box liveness: 1 when the box can reach a goal
  valid: boolean; // false when boxes and goals do not match
}

/** Shortest walk (BFS) on the whole level avoiding boxes; null if none. */
function bfsPathGlobal(
  from: number,
  to: number,
  level: Level,
  boxes: ReadonlySet<number>,
): string | null {
  if (from === to) return '';
  const N = level.w * level.h;
  if (boxes.has(to)) return null;
  const parent = new Int32Array(N).fill(-1);
  const pdir = new Int8Array(N);
  const q: number[] = [from];
  parent[from] = from;
  let head = 0;
  while (head < q.length) {
    const u = q[head]!;
    head++;
    const ns = globalNeighbours(u, level.w, level.h);
    for (let d = 0; d < 4; d++) {
      const v = ns[d]!;
      if (v < 0 || !level.floor[v] || boxes.has(v) || parent[v] >= 0) continue;
      parent[v] = u;
      pdir[v] = d;
      if (v === to) {
        let s = '';
        let cur = v;
        while (cur !== from) {
          s = DIRS[pdir[cur]!]!.ch + s;
          cur = parent[cur]!;
        }
        return s;
      }
      q.push(v);
    }
  }
  return null;
}

function buildRoom(level: Level, roomCells: number[], door: number): Room {
  const boxSet = new Set(level.boxes);
  // Solver floor: the room plus the floor neighbours of the door that lie
  // outside it (the passage to the hallway). A neighbour holding a box
  // belongs to another room and stays put while this room is solved, so it
  // is treated as a wall here, never as floor.
  const floorSet = new Set(roomCells);
  for (const nb of globalNeighbours(door, level.w, level.h)) {
    if (nb >= 0 && level.floor[nb] && !floorSet.has(nb) && !boxSet.has(nb)) {
      floorSet.add(nb);
    }
  }
  const cells = [...floorSet].sort((a, b) => a - b);
  const n = cells.length;
  const local = new Int32Array(level.w * level.h).fill(-1);
  cells.forEach((c, i) => {
    local[c] = i;
  });
  const neigh = new Int32Array(n * 4).fill(-1);
  const cellX = new Int32Array(n);
  const cellY = new Int32Array(n);
  cells.forEach((c, i) => {
    cellX[i] = c % level.w;
    cellY[i] = (c / level.w) | 0;
    const ns = globalNeighbours(c, level.w, level.h);
    for (let d = 0; d < 4; d++) {
      const v = ns[d]!;
      if (v >= 0) neigh[i * 4 + d] = local[v];
    }
  });
  const boxes0: number[] = [];
  const goals: number[] = [];
  const boxesGlobal: number[] = [];
  const goalsGlobal: number[] = [];
  for (const c of cells) {
    if (boxSet.has(c)) {
      boxes0.push(local[c]);
      boxesGlobal.push(c);
    }
    if (level.goal[c]) {
      goals.push(local[c]);
      goalsGlobal.push(c);
    }
  }
  boxes0.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  // Entry: the door itself, unless a box sits on the door — then the
  // player starts on the free passage cell beside it (closest to the
  // level's player start, ties broken by cell index) and pushes in.
  let entry = door;
  if (boxSet.has(door)) {
    const options: number[] = [];
    for (const nb of globalNeighbours(door, level.w, level.h)) {
      if (nb >= 0 && local[nb] >= 0 && !boxSet.has(nb)) options.push(nb);
    }
    if (options.length > 0) {
      let bestCell = options[0]!;
      let bestLen = Infinity;
      for (const o of options) {
        const path = bfsPathGlobal(level.player, o, level, boxSet);
        const len = path === null ? Infinity : path.length;
        if (len < bestLen || (len === bestLen && o < bestCell)) {
          bestLen = len;
          bestCell = o;
        }
      }
      entry = bestCell;
    }
  }
  const goalSet = new Uint8Array(n);
  for (const g of goals) goalSet[g] = 1;
  const goalCode = encodeBoxes(goals, n);
  // All-pairs shortest paths ignoring boxes.
  const dist = new Int32Array(n * n).fill(-1);
  for (let s = 0; s < n; s++) {
    const q: number[] = [s];
    dist[s * n + s] = 0;
    let head = 0;
    while (head < q.length) {
      const u = q[head]!;
      head++;
      for (let d = 0; d < 4; d++) {
        const v = neigh[u * 4 + d]!;
        if (v >= 0 && dist[s * n + v] < 0) {
          dist[s * n + v] = dist[s * n + u] + 1;
          q.push(v);
        }
      }
    }
  }
  const room: Room = {
    w: level.w,
    h: level.h,
    door,
    entry,
    cells,
    local,
    n,
    neigh,
    cellX,
    cellY,
    goals,
    goalSet,
    goalCode,
    boxes0,
    goalsGlobal,
    boxesGlobal,
    entryLocal: local[entry],
    dist,
    live: new Uint8Array(n),
    valid: boxes0.length === goals.length,
  };
  room.live = computeLive(room);
  return room;
}

function encodeBoxes(boxes: number[], n: number): number {
  let code = 0;
  for (const b of boxes) code = code * n + b;
  return code;
}

function decodeBoxes(code: number, k: number, n: number): number[] {
  const boxes = new Array<number>(k);
  let tmp = code;
  for (let i = k - 1; i >= 0; i--) {
    boxes[i] = tmp % n;
    tmp = (tmp / n) | 0;
  }
  return boxes;
}

/**
 * Exact single-box liveness: a cell is live when a lone box there can be
 * pushed onto a goal. Computed by reverse BFS over (box, player) pairs
 * from every goal state, following walks backwards and pulls backwards.
 */
function computeLive(room: Room): Uint8Array {
  const n = room.n;
  const seen = new Uint8Array(n * n);
  const queue: number[] = [];
  for (const g of room.goals) {
    for (let p = 0; p < n; p++) {
      if (p === g) continue;
      const s = g * n + p;
      if (!seen[s]) {
        seen[s] = 1;
        queue.push(s);
      }
    }
  }
  let head = 0;
  while (head < queue.length) {
    const s = queue[head]!;
    head++;
    const b = (s / n) | 0;
    const p = s % n;
    for (let d = 0; d < 4; d++) {
      const p2 = room.neigh[p * 4 + d]!;
      if (p2 >= 0 && p2 !== b) {
        const s2 = b * n + p2;
        if (!seen[s2]) {
          seen[s2] = 1;
          queue.push(s2);
        }
      }
      // Pull predecessor: this state came from pushing the box from p
      // (player beside the box) with the player one step further out.
      if (room.neigh[b * 4 + d] === p) {
        const prevPlayer = room.neigh[p * 4 + d]!;
        if (prevPlayer >= 0 && prevPlayer !== p) {
          const s2 = p * n + prevPlayer;
          if (!seen[s2]) {
            seen[s2] = 1;
            queue.push(s2);
          }
        }
      }
    }
  }
  const live = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    for (let p = 0; p < n; p++) {
      if (p !== c && seen[c * n + p]) {
        live[c] = 1;
        break;
      }
    }
  }
  return live;
}

function detectRooms(level: Level): Room[] {
  if (level.boxes.length === 0) return [];
  for (const range of [
    [3, 4],
    [1, 6],
  ] as const) {
    const cands = collectCandidates(level, range[0], range[1]);
    const part = partitionRooms(level, cands);
    if (part) {
      const rooms = part.map((c) => buildRoom(level, c.roomCells, c.door));
      if (rooms.every((r) => r.valid)) return rooms;
      log('room partition rejected: a room has boxes != goals');
    }
  }
  // Last resort: the whole level as one room.
  const allFloor: number[] = [];
  for (let i = 0; i < level.w * level.h; i++) if (level.floor[i]) allFloor.push(i);
  return [buildRoom(level, allFloor, level.player)];
}

// ------------------------------------------------------------ room A* -----

interface RoomResult {
  moves: string;
  pushes: number;
  expansions: number;
  endPlayer: number; // global cell where the player ends up
  endPlayerLocal: number;
  finalBoxes: number[]; // local box cells at the end
}

interface SharedBudget {
  used: number;
  cap: number;
}

function allPerms(k: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Uint8Array(k);
  const rec = (): void => {
    if (cur.length === k) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < k; i++) {
      if (used[i]) continue;
      used[i] = 1;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = 0;
    }
  };
  rec();
  return out;
}

/** Cells reachable from `start` avoiding `boxes`; also the smallest one. */
function reachSet(
  room: Room,
  start: number,
  boxes: number[],
): { set: Uint8Array; min: number } {
  const n = room.n;
  const occ = new Uint8Array(n);
  for (const b of boxes) occ[b] = 1;
  const set = new Uint8Array(n);
  const q: number[] = [start];
  set[start] = 1;
  let min = start;
  let head = 0;
  while (head < q.length) {
    const u = q[head]!;
    head++;
    for (let d = 0; d < 4; d++) {
      const v = room.neigh[u * 4 + d]!;
      if (v >= 0 && !occ[v] && !set[v]) {
        set[v] = 1;
        if (v < min) min = v;
        q.push(v);
      }
    }
  }
  return { set, min };
}

/** Shortest walk inside the room avoiding boxes; null if none. */
function bfsPathLocal(
  room: Room,
  from: number,
  to: number,
  boxes: number[],
): string | null {
  if (from === to) return '';
  const n = room.n;
  const occ = new Uint8Array(n);
  for (const b of boxes) occ[b] = 1;
  if (occ[to]) return null;
  const parent = new Int32Array(n).fill(-1);
  const pdir = new Int8Array(n);
  const q: number[] = [from];
  parent[from] = from;
  let head = 0;
  while (head < q.length) {
    const u = q[head]!;
    head++;
    for (let d = 0; d < 4; d++) {
      const v = room.neigh[u * 4 + d]!;
      if (v < 0 || occ[v] || parent[v] >= 0) continue;
      parent[v] = u;
      pdir[v] = d;
      if (v === to) {
        let s = '';
        let cur = v;
        while (cur !== from) {
          s = DIRS[pdir[cur]!]!.ch + s;
          cur = parent[cur]!;
        }
        return s;
      }
      q.push(v);
    }
  }
  return null;
}

/**
 * Sound 2x2 freeze prune: a box (not on a goal) inside a 2x2 square of
 * wall-or-box cells can never move again — every push out needs the player
 * to stand on a block cell, every push in needs a free block cell — and no
 * outside box can enter the block. Cells outside the room floor count as
 * walls: pushes never leave the room.
 */
function hasFrozenBox(room: Room, boxes: number[], occ: Uint8Array): boolean {
  for (const b of boxes) {
    if (room.goalSet[b]) continue;
    const gx = room.cellX[b]!;
    const gy = room.cellY[b]!;
    for (let k = 0; k < 4; k++) {
      const tx = gx - (k === 1 || k === 3 ? 1 : 0);
      const ty = gy - (k >= 2 ? 1 : 0);
      let blocked = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const cx = tx + dx;
          const cy = ty + dy;
          if (cx < 0 || cy < 0 || cx >= 15 + 1 || cy < 0) {
            blocked++;
            continue;
          }
          const g = cy * 1000000 + cx; // placeholder, replaced below
          void g;
          blocked += 0;
        }
      }
      void blocked;
    }
  }
  return false;
}

// ------------------------------------------------------------ room A* -----

/**
 * Push-optimal A* for one room (weight 1), or weighted A* (weight > 1)
 * for a fast valid-but-not-necessarily-optimal fallback. A state is the
 * sorted box positions plus the canonical player position; the goal is
 * every box on a goal and, when requireExit, the player able to walk back
 * to the entry cell. Returns null when the node budget runs out.
 */
function solveRoom(
  room: Room,
  requireExit: boolean,
  weight: number,
  budget: number,
  shared: SharedBudget,
): RoomResult | null {
  const n = room.n;
  const k = room.boxes0.length;
  if (k === 0) {
    return {
      moves: '',
      pushes: 0,
      expansions: 0,
      endPlayer: room.entry,
      endPlayerLocal: room.entryLocal,
      finalBoxes: [],
    };
  }
  if (!room.valid) return null;
  const perms = k <= 5 ? allPerms(k) : null;
  const hCache = new Map<number, number>();
  const heuristic = (boxCode: number): number => {
    const cached = hCache.get(boxCode);
    if (cached !== undefined) return cached;
    const boxes = decodeBoxes(boxCode, k, n);
    let best = Infinity;
    if (perms !== null) {
      for (const perm of perms) {
        let sum = 0;
        let ok = true;
        for (let i = 0; i < k; i++) {
          const d = room.dist[boxes[i]! * n + room.goals[perm[i]!]!]!;
          if (d < 0) {
            ok = false;
            break;
          }
          sum += d;
          if (sum >= best) {
            ok = false;
            break;
          }
        }
        if (ok && sum < best) best = sum;
      }
    } else {
      // Many boxes (whole-level fallback): sum of each box's nearest goal.
      let sum = 0;
      for (let i = 0; i < k; i++) {
        let m = Infinity;
        for (const g of room.goals) {
          const d = room.dist[boxes[i]! * n + g]!;
          if (d >= 0 && d < m) m = d;
        }
        if (m === Infinity) {
          sum = Infinity;
          break;
        }
        sum += m;
      }
      best = sum;
    }
    const h = best === Infinity ? -1 : best;
    hCache.set(boxCode, h);
    return h;
  };

  // Binary heap on (f, h, insertion sequence), parallel arrays.
  const heapF: number[] = [];
  const heapH: number[] = [];
  const heapSeq: number[] = [];
  const heapId: number[] = [];
  let seq = 0;
  const less = (i: number, j: number): boolean =>
    heapF[i]! < heapF[j]! ||
    (heapF[i] === heapF[j] &&
      (heapH[i]! < heapH[j]! ||
        (heapH[i] === heapH[j] && heapSeq[i]! < heapSeq[j]!)));
  const swap = (i: number, j: number): void => {
    let t = heapF[i]!;
    heapF[i] = heapF[j]!;
    heapF[j] = t;
    t = heapH[i]!;
    heapH[i] = heapH[j]!;
    heapH[j] = t;
    t = heapSeq[i]!;
    heapSeq[i] = heapSeq[j]!;
    heapSeq[j] = t;
    t = heapId[i]!;
    heapId[i] = heapId[j]!;
    heapId[j] = t;
  };
  const heapPush = (f: number, h: number, id: number): void => {
    heapF.push(f);
    heapH.push(h);
    heapSeq.push(seq++);
    heapId.push(id);
    let i = heapF.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      swap(i, p);
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heapId[0]!;
    const last = heapId.length - 1;
    heapF[0] = heapF[last]!;
    heapH[0] = heapH[last]!;
    heapSeq[0] = heapSeq[last]!;
    heapId[0] = heapId[last]!;
    heapF.pop();
    heapH.pop();
    heapSeq.pop();
    heapId.pop();
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= heapId.length) break;
      if (c + 1 < heapId.length && less(c + 1, c)) c++;
      if (!less(c, i)) break;
      swap(i, c);
      i = c;
    }
    return top;
  };

  const sBox: number[] = [];
  const sCanon: number[] = [];
  const sG: number[] = [];
  const sParent: number[] = [];
  const sPushBox: number[] = [];
  const sPushDir: number[] = [];
  const best = new Map<number, number>();

  const startCode = encodeBoxes(room.boxes0, n);
  const startReach = reachSet(room, room.entryLocal, room.boxes0);
  const occ0 = new Uint8Array(n);
  for (const b of room.boxes0) occ0[b] = 1;
  if (room.boxes0.some((b) => !room.live[b])) return null;
  if (hasFrozenBox(room, room.boxes0, occ0)) return null;
  const h0 = heuristic(startCode);
  if (h0 < 0) return null;
  sBox.push(startCode);
  sCanon.push(startReach.min);
  sG.push(0);
  sParent.push(-1);
  sPushBox.push(-1);
  sPushDir.push(-1);
  best.set(startCode * n + startReach.min, 0);
  heapPush(h0, h0, 0);

  let expansions = 0;
  while (heapId.length > 0) {
    if (expansions >= budget || shared.used >= shared.cap) return null;
    const id = heapPop();
    const g = sG[id]!;
    const boxCode = sBox[id]!;
    const canon = sCanon[id]!;
    if (best.get(boxCode * n + canon) !== g) continue; // stale entry
    expansions++;
    shared.used++;
    const boxes = decodeBoxes(boxCode, k, n);
    const R = reachSet(room, canon, boxes);
    if (boxCode === room.goalCode && (!requireExit || R.set[room.entryLocal])) {
      // Reconstruct: replay the pushes, walking between them.
      const pushes: { box: number; dir: number }[] = [];
      let cur = id;
      while (sParent[cur]! >= 0) {
        pushes.push({ box: sPushBox[cur]!, dir: sPushDir[cur]! });
        cur = sParent[cur]!;
      }
      pushes.reverse();
      let curPlayer = room.entryLocal;
      let curBoxes = [...room.boxes0];
      let moves = '';
      for (const p of pushes) {
        const behind = room.neigh[p.box * 4 + OPP[p.dir]!]!;
        const path = bfsPathLocal(room, curPlayer, behind, curBoxes);
        if (path === null) return null;
        moves += path + DIRS[p.dir]!.ch;
        const dest = room.neigh[p.box * 4 + p.dir]!;
        const idx = curBoxes.indexOf(p.box);
        curBoxes[idx] = dest;
        curBoxes.sort((a, b) => a - b);
        curPlayer = p.box;
      }
      if (requireExit) {
        const exitPath = bfsPathLocal(room, curPlayer, room.entryLocal, curBoxes);
        if (exitPath === null) return null;
        moves += exitPath;
        curPlayer = room.entryLocal;
      }
      return {
        moves,
        pushes: pushes.length,
        expansions,
        endPlayer: room.cells[curPlayer]!,
        endPlayerLocal: curPlayer,
        finalBoxes: curBoxes,
      };
    }
    const occ = new Uint8Array(n);
    for (const b of boxes) occ[b] = 1;
    for (let i = 0; i < k; i++) {
      const b = boxes[i]!;
      for (let d = 0; d < 4; d++) {
        const dest = room.neigh[b * 4 + d]!;
        if (dest < 0 || occ[dest]) continue;
        const behind = room.neigh[b * 4 + OPP[d]!]!;
        if (behind < 0 || !R.set[behind]) continue;
        const nb = [...boxes];
        nb[i] = dest;
        nb.sort((a, b2) => a - b2);
        let dead = false;
        for (const x of nb) {
          if (!room.live[x]) {
            dead = true;
            break;
          }
        }
        if (dead) continue;
        const nOcc = new Uint8Array(n);
        for (const x of nb) nOcc[x] = 1;
        if (hasFrozenBox(room, nb, nOcc)) continue;
        const nCode = encodeBoxes(nb, n);
        const h = heuristic(nCode);
        if (h < 0) continue;
        const g2 = g + 1;
        const R2 = reachSet(room, b, nb);
        const key2 = nCode * n + R2.min;
        const prev = best.get(key2);
        if (prev !== undefined && prev <= g2) continue;
        best.set(key2, g2);
        const nid = sBox.length;
        sBox.push(nCode);
        sCanon.push(R2.min);
        sG.push(g2);
        sParent.push(id);
        sPushBox.push(b);
        sPushDir.push(d);
        heapPush(g2 + weight * h, h, nid);
      }
    }
  }
  return null;
}

function solveRoomWithFallbacks(
  room: Room,
  requireExit: boolean,
  shared: SharedBudget,
): RoomResult | null {
  const remaining = (): number => Math.max(0, shared.cap - shared.used);
  const attempts: { w: number; b: number; exit: boolean }[] = requireExit
    ? [
        { w: 1, b: 500_000, exit: true },
        { w: 2, b: 500_000, exit: true },
        { w: 3, b: 1_500_000, exit: true },
        { w: 1, b: 500_000, exit: false },
      ]
    : [
        { w: 1, b: 500_000, exit: false },
        { w: 2, b: 500_000, exit: false },
        { w: 3, b: 1_500_000, exit: false },
      ];
  for (const a of attempts) {
    if (remaining() <= 0) return null;
    const res = solveRoom(room, a.exit, a.w, Math.min(a.b, remaining()), shared);
    if (res === null) continue;
    if (!a.exit && requireExit) {
      // Solved push-optimally without the exit condition: use it only
      // when the player can still walk back out to the entry cell.
      const exitPath = bfsPathLocal(
        room,
        res.endPlayerLocal,
        room.entryLocal,
        res.finalBoxes,
      );
      if (exitPath === null) continue;
      return {
        ...res,
        moves: res.moves + exitPath,
        endPlayer: room.entry,
        endPlayerLocal: room.entryLocal,
      };
    }
    return res;
  }
  return null;
}

// -------------------------------------------------------------- stitching ---

function solveLevel(level: Level, shared: SharedBudget): string {
  const rooms = detectRooms(level);
  log(
    `rooms: ${rooms
      .map(
        (r) =>
          `door=${r.door} entry=${r.entry} cells=${r.n} boxes=${r.boxes0.length} goals=${r.goals.length} valid=${r.valid}`,
      )
      .join(' | ')}`,
  );
  // Deterministic room order: left to right along the hallway, then top
  // to bottom (doors sit in distinct columns).
  rooms.sort(
    (a, b) =>
      a.door % level.w - (b.door % level.w) ||
      ((a.door / level.w) | 0) - ((b.door / level.w) | 0),
  );
  let moves = '';
  let curPlayer = level.player;
  const curBoxes = new Set<number>(level.boxes);
  const remaining = [...rooms];
  while (remaining.length > 0) {
    let picked = -1;
    let walk: string | null = null;
    for (let i = 0; i < remaining.length; i++) {
      const w = bfsPathGlobal(curPlayer, remaining[i]!.entry, level, curBoxes);
      if (w !== null) {
        picked = i;
        walk = w;
        break;
      }
    }
    if (picked < 0) {
      log('stuck: no remaining room is reachable');
      break;
    }
    const room = remaining.splice(picked, 1)[0]!;
    moves += walk!;
    curPlayer = room.entry;
    const last = remaining.length === 0;
    let res: RoomResult | null = null;
    try {
      res = solveRoomWithFallbacks(room, !last, shared);
    } catch (e) {
      log(`room door=${room.door} threw: ${e}`);
      res = null;
    }
    if (res === null) {
      log(`room door=${room.door} unsolved; level left partial`);
      break;
    }
    log(
      `room door=${room.door}: solved in ${res.pushes} pushes, ${res.expansions} expansions`,
    );
    moves += res.moves;
    curPlayer = res.endPlayer;
    for (const b of room.boxesGlobal) curBoxes.delete(b);
    for (const g of room.goalsGlobal) curBoxes.add(g);
  }
  return moves;
}

// ------------------------------------------------------------------ main ---

const shared: SharedBudget = { used: 0, cap: 6_000_000 };
for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
    const level = parseLevel(text);
    moves = solveLevel(level, shared);
    log(`${id}: ${moves.length} moves`);
  } catch (e) {
    console.error(`error solving ${id}: ${e}`);
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}
