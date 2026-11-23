// Data from experiments/demo_walk.py and the canvas drawing of one frame.

export type Seg = [number, number, number, number, string | null] // x0, y0, x1, y1, true wall or null

export type Run = {
  fused: Seg[][] // walls after each position
  cand: [number, number][][] // candidate lines (theta, rho) of each position
  found: [number, number][][] // paths of each position: s = cos(phi), path length
}

export type Walk = {
  plan: { width: number; depth: number; walls: [number, number, number, number][]; rooms: [string, number, number][] }
  ap: [number, number]
  targets: string[]
  positions: [number, number][]
  truth: [number, number, number][][] // s, path length, order
  bws: number[]
  runs: Record<string, Record<string, Run>>
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
export function drawFrame(canvas: HTMLCanvasElement, w: Walk, run: Run, t: number) {
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
  drawPlan(ctx, w, run, t, n, 0, 0, pw, ch, k)
  if (side) drawPaths(ctx, w, run, n, pw + 24, 0, cw - pw - 24, ch, k)
}

function drawPlan(ctx: CanvasRenderingContext2D, w: Walk, run: Run, t: number, n: number, x0: number, y0: number, pw: number, ph: number, k: Ink) {
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
  const cur = run.cand[n - 1] ?? []
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

  // fused walls
  ctx.lineWidth = 4
  for (const [a, b, c, d, lab] of run.fused[n - 1] ?? []) {
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
    if (L > lmax) continue
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
