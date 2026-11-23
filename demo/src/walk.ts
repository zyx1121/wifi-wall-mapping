// Data from experiments/demo_walk.py and the canvas drawing of one frame.

export type Seg = [number, number, number, number, string | null] // x0, y0, x1, y1, true wall or null

export type Snap = {
  n: number // positions used
  rooms: number[] // room label per grid cell, run-length coded: value, count, ...
  rooms_ok: number
  walls: [number, number, number, number][] // closed walls
  heat: number[] // predicted power per heat cell (dBm)
  heat_err: number // median error behind walls (dB)
}

export type Run = {
  fused: Seg[][] // walls after each position
  cand: [number, number][][] // candidate lines (theta, rho) of each position
  found: [number, number][][] // paths of each position: s = cos(phi), path length
  snaps: Snap[]
}

type Grid = { x0: number; y0: number; cell: number; nx: number; ny: number }

export type Layer = "walls" | "rooms" | "heat"

export type Walk = {
  plan: { width: number; depth: number; walls: [number, number, number, number][]; rooms: [string, number, number][] }
  ap: [number, number]
  targets: string[]
  positions: [number, number][]
  truth: [number, number, number][][] // s, path length, order
  bws: number[]
  runs: Record<string, Record<string, Run>>
  rooms_grid: Grid
  heat_grid: Grid
  heat_truth: number[]
}

export function snapAt(run: Run, n: number): Snap | null {
  let out: Snap | null = null
  for (const s of run.snaps) if (s.n <= n) out = s
  return out
}

export const HEAT_MIN = -80
export const HEAT_MAX = -30
// viridis stops, for data only
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]

export function heatColor(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, (v - HEAT_MIN) / (HEAT_MAX - HEAT_MIN))) * (VIRIDIS.length - 1)
  const i = Math.min(VIRIDIS.length - 2, Math.floor(t))
  const f = t - i
  return [0, 1, 2].map(k => Math.round(VIRIDIS[i][k] * (1 - f) + VIRIDIS[i + 1][k] * f)) as [number, number, number]
}

const ROOM_COLORS = [[96, 165, 250], [52, 211, 153], [251, 191, 36], [244, 114, 182], [167, 139, 250], [45, 212, 191], [251, 146, 60], [148, 163, 184]]

const decoded = new WeakMap<Snap, Uint8Array>()

function roomCells(s: Snap, g: Grid): Uint8Array {
  let out = decoded.get(s)
  if (!out) {
    out = new Uint8Array(g.nx * g.ny)
    let k = 0
    for (let i = 0; i < s.rooms.length; i += 2) {
      out.fill(s.rooms[i], k, k + s.rooms[i + 1])
      k += s.rooms[i + 1]
    }
    decoded.set(s, out)
  }
  return out
}

const raster = typeof document !== "undefined" ? document.createElement("canvas") : null

// a grid image, row 0 at the bottom, drawn into the plan box
function gridImage(ctx: CanvasRenderingContext2D, g: Grid, color: (k: number) => [number, number, number, number], X: (x: number) => number, Y: (y: number) => number, smooth: boolean) {
  if (!raster) return
  raster.width = g.nx
  raster.height = g.ny
  const rc = raster.getContext("2d")!
  const img = rc.createImageData(g.nx, g.ny)
  for (let iy = 0; iy < g.ny; iy++) {
    for (let ix = 0; ix < g.nx; ix++) {
      const c = color(iy * g.nx + ix)
      const o = ((g.ny - 1 - iy) * g.nx + ix) * 4
      img.data[o] = c[0]
      img.data[o + 1] = c[1]
      img.data[o + 2] = c[2]
      img.data[o + 3] = c[3]
    }
  }
  rc.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = smooth
  ctx.drawImage(raster, X(g.x0), Y(g.y0 + g.ny * g.cell), X(g.x0 + g.nx * g.cell) - X(g.x0), Y(g.y0) - Y(g.y0 + g.ny * g.cell))
}

export const METHODS = [
  { id: "ifft", name: "IFFT" },
  { id: "omp", name: "OMP" },
  { id: "nomp", name: "NOMP" },
  { id: "music", name: "MUSIC" },
  { id: "mpencil", name: "Matrix Pencil" },
] as const

const C = { found: "#f5a524", hit: "#2fbf71", miss: "#ef4f4f" }

export function score(segs: Seg[], targets: string[]) {
  const hit = new Set(segs.map(s => s[4]).filter((l): l is string => !!l && targets.includes(l)))
  return { found: hit.size, falses: segs.filter(s => s[4] === null).length }
}

type Ink = { fg: string; muted: string; faint: string }

function ink(): Ink {
  const cs = getComputedStyle(document.documentElement)
  return { fg: cs.getPropertyValue("--foreground").trim(), muted: cs.getPropertyValue("--muted-foreground").trim(), faint: cs.getPropertyValue("--border").trim() }
}

const FONT = "14px Inter, 'Noto Sans TC', sans-serif"

/** Draw one frame. t is the playhead in positions (fractional). */
export function drawFrame(canvas: HTMLCanvasElement, w: Walk, run: Run, t: number, layer: Layer = "walls") {
  t = Math.max(0, Math.min(w.positions.length - 1, t))
  const dpr = window.devicePixelRatio || 1
  const cw = canvas.clientWidth
  const ch = canvas.clientHeight
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr)
    canvas.height = Math.round(ch * dpr)
  }
  const ctx = canvas.getContext("2d")!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cw, ch)
  const k = ink()

  // split: plan on the left, current paths on the right when there is room
  const side = cw >= 720 && cw >= ch * 1.35
  const pw = side ? cw * 0.68 : cw
  const n = Math.max(1, Math.min(w.positions.length, Math.floor(t) + 1))
  drawPlan(ctx, w, run, t, n, 0, 0, pw, ch, k, layer)
  if (side) drawPaths(ctx, w, run, n, pw + 24, 0, cw - pw - 24, ch, k)
}

function drawPlan(ctx: CanvasRenderingContext2D, w: Walk, run: Run, t: number, n: number, x0: number, y0: number, pw: number, ph: number, k: Ink, layer: Layer) {
  const bx = [-1.2, w.plan.width + 1.2]
  const by = [-0.8, w.plan.depth + 0.8]
  const sc = Math.min(pw / (bx[1] - bx[0]), ph / (by[1] - by[0]))
  const ox = x0 + (pw - sc * (bx[1] - bx[0])) / 2
  const oy = y0 + (ph - sc * (by[1] - by[0])) / 2
  const X = (x: number) => ox + (x - bx[0]) * sc
  const Y = (y: number) => oy + (by[1] - y) * sc

  ctx.save()
  ctx.beginPath()
  ctx.rect(X(bx[0]), Y(by[1]), (bx[1] - bx[0]) * sc, (by[1] - by[0]) * sc)
  ctx.clip()

  const snap = layer === "walls" ? null : snapAt(run, n)
  if (snap && layer === "heat") {
    gridImage(ctx, w.heat_grid, i => [...heatColor(snap.heat[i]), 255], X, Y, true)
  }
  if (snap && layer === "rooms") {
    const cells = roomCells(snap, w.rooms_grid)
    gridImage(ctx, w.rooms_grid, i => (cells[i] ? [...ROOM_COLORS[(cells[i] - 1) % ROOM_COLORS.length], 90] as [number, number, number, number] : [0, 0, 0, 0]), X, Y, false)
  }

  // floor plan
  ctx.lineCap = "round"
  ctx.strokeStyle = k.faint
  ctx.lineWidth = 6
  for (const [a, b, c, d] of w.plan.walls) {
    ctx.beginPath()
    ctx.moveTo(X(a), Y(b))
    ctx.lineTo(X(c), Y(d))
    ctx.stroke()
  }
  ctx.font = FONT
  ctx.textAlign = "center"
  ctx.fillStyle = k.muted
  for (const [name, x, y] of w.plan.rooms) ctx.fillText(name, X(x), Y(y))

  // walk so far
  ctx.strokeStyle = k.muted
  ctx.globalAlpha = 0.5
  ctx.lineWidth = 1.5
  ctx.setLineDash([2, 4])
  ctx.beginPath()
  w.positions.slice(0, n).forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))))
  ctx.stroke()
  ctx.setLineDash([])
  ctx.globalAlpha = 1

  // candidate walls of the current position
  const cur = layer === "walls" ? (run.cand[n - 1] ?? []) : []
  const len = (bx[1] - bx[0]) + (by[1] - by[0])
  ctx.strokeStyle = C.found
  ctx.globalAlpha = 0.45
  ctx.lineWidth = 1.5
  for (const [th, rho] of cur) {
    const nx = Math.cos(th)
    const ny = Math.sin(th)
    const fx = w.ap[0] + rho * nx
    const fy = w.ap[1] + rho * ny
    ctx.beginPath()
    ctx.moveTo(X(fx + ny * len), Y(fy - nx * len))
    ctx.lineTo(X(fx - ny * len), Y(fy + nx * len))
    ctx.stroke()
  }
  ctx.globalAlpha = 1

  // closed walls of the room map
  if (snap) {
    ctx.save()
    ctx.beginPath()
    ctx.rect(X(0) - 2, Y(w.plan.depth) - 2, X(w.plan.width) - X(0) + 4, Y(0) - Y(w.plan.depth) + 4)
    ctx.clip()
    ctx.strokeStyle = k.fg
    ctx.lineWidth = 2
    for (const [a, b, c, d] of snap.walls) {
      ctx.beginPath()
      ctx.moveTo(X(a), Y(b))
      ctx.lineTo(X(c), Y(d))
      ctx.stroke()
    }
    ctx.restore()
  }

  // fused walls
  ctx.lineWidth = 4
  for (const [a, b, c, d, lab] of layer === "walls" ? (run.fused[n - 1] ?? []) : []) {
    ctx.strokeStyle = lab ? C.hit : C.miss
    ctx.beginPath()
    ctx.moveTo(X(a), Y(b))
    ctx.lineTo(X(c), Y(d))
    ctx.stroke()
  }

  // AP and the walker
  ctx.fillStyle = k.fg
  ctx.fillRect(X(w.ap[0]) - 7, Y(w.ap[1]) - 7, 14, 14)
  ctx.textAlign = "left"
  ctx.fillStyle = k.muted
  ctx.fillText("AP", X(w.ap[0]) + 12, Y(w.ap[1]) - 8)
  const i = Math.min(w.positions.length - 1, Math.floor(t))
  const j = Math.min(w.positions.length - 1, i + 1)
  const f = t - Math.floor(t)
  const px = w.positions[i][0] * (1 - f) + w.positions[j][0] * f
  const py = w.positions[i][1] * (1 - f) + w.positions[j][1] * f
  ctx.fillStyle = k.fg
  ctx.beginPath()
  ctx.arc(X(px), Y(py), 8, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function drawPaths(ctx: CanvasRenderingContext2D, w: Walk, run: Run, n: number, x0: number, y0: number, pw: number, ph: number, k: Ink) {
  const pad = { l: 36, r: 16, b: 40, t: 28 }
  const gw = pw - pad.l - pad.r
  const gh = Math.min(ph - pad.b - pad.t, gw * 1.4)
  const top = y0 + pad.t + (ph - pad.b - pad.t - gh) / 2
  const lmax = 20
  const X = (s: number) => x0 + pad.l + ((s + 1) / 2) * gw
  const Y = (L: number) => top + gh - (Math.min(L, lmax) / lmax) * gh

  ctx.font = FONT
  ctx.fillStyle = k.muted
  ctx.textAlign = "left"
  ctx.fillText("這個位置的路徑", x0, top - 12)
  ctx.strokeStyle = k.faint
  ctx.lineWidth = 1
  ctx.strokeRect(X(-1), Y(lmax), gw, gh)

  for (const [s, L, order] of w.truth[n - 1] ?? []) {
    ctx.strokeStyle = order === 0 ? k.fg : k.muted
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(X(s), Y(L), order === 0 ? 7 : 6, 0, Math.PI * 2)
    ctx.stroke()
  }
  ctx.strokeStyle = C.found
  ctx.lineWidth = 2.5
  for (const [s, L] of run.found[n - 1] ?? []) {
    if (L > lmax || L < 0) continue
    ctx.beginPath()
    ctx.moveTo(X(s) - 5, Y(L) - 5)
    ctx.lineTo(X(s) + 5, Y(L) + 5)
    ctx.moveTo(X(s) + 5, Y(L) - 5)
    ctx.lineTo(X(s) - 5, Y(L) + 5)
    ctx.stroke()
  }

  ctx.fillStyle = k.muted
  ctx.textAlign = "center"
  for (const deg of [180, 90, 0]) ctx.fillText(`${deg}°`, X(Math.cos((deg * Math.PI) / 180)), top + gh + 18)
  ctx.fillText("到達角", X(0), top + gh + 36)
  ctx.textAlign = "right"
  for (const L of [0, 10, 20]) ctx.fillText(`${L} m`, x0 + pad.l - 6, Y(L) + 5)
  ctx.textAlign = "left"
}
