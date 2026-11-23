// Canvas drawing for the room view and the angle-delay map.
// Map axes: x = s = cos(phi) (west on the left, east on the right, like the
// room), y = path length from the AP (0 at the bottom).

import { bytes, foundAt, type Final, type MapData, type Pt, type Scene, type Step } from "./trace"

export const DATA = {
  found: "#f5a524",
  hit: "#2fbf71",
  miss: "#ef4f4f",
}

export type Ink = { fg: string; muted: string; faint: string; bg: string }

export function readInk(): Ink {
  const cs = getComputedStyle(document.documentElement)
  const v = (n: string) => cs.getPropertyValue(n).trim()
  return { fg: v("--foreground"), muted: v("--muted-foreground"), faint: v("--border"), bg: v("--background") }
}

export function setup(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
  }
  const ctx = canvas.getContext("2d")!
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  return { ctx, w, h }
}

const FONT = "14px Inter, 'Noto Sans TC', sans-serif"

// ------------------------------------------------------------ room view

export function drawRoom(canvas: HTMLCanvasElement, sc: Scene, step: Step, final: Final, ink: Ink) {
  const { ctx, w, h } = setup(canvas)
  const [W, D] = sc.room
  const bx: [number, number] = [-W * 0.9, W * 1.9]
  const by: [number, number] = [-D * 0.5, D * 1.9]
  const scale = Math.min((w - 16) / (bx[1] - bx[0]), (h - 16) / (by[1] - by[0]))
  const ox = (w - scale * (bx[1] - bx[0])) / 2
  const oy = (h - scale * (by[1] - by[0])) / 2
  const X = (x: number) => ox + (x - bx[0]) * scale
  const Y = (y: number) => h - oy - (y - by[0]) * scale

  ctx.save()
  ctx.beginPath()
  ctx.rect(X(bx[0]), Y(by[1]), (bx[1] - bx[0]) * scale, (by[1] - by[0]) * scale)
  ctx.clip()

  // mirror rooms: where first-order images live
  ctx.strokeStyle = ink.faint
  ctx.setLineDash([4, 4])
  ctx.lineWidth = 1
  for (const [dx, dy] of [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [1, 1]]) {
    ctx.strokeRect(X(dx * W), Y((dy + 1) * D), W * scale, D * scale)
  }
  ctx.setLineDash([])

  // walls
  ctx.strokeStyle = ink.fg
  ctx.lineWidth = 2
  for (const wl of sc.walls) {
    ctx.beginPath()
    ctx.moveTo(X(wl.p0[0]), Y(wl.p0[1]))
    ctx.lineTo(X(wl.p1[0]), Y(wl.p1[1]))
    ctx.stroke()
  }

  // true images
  ctx.font = FONT
  for (const p of sc.paths) {
    if (p.order === 0) continue
    const r = p.order === 1 ? 6 : 4
    ctx.strokeStyle = p.order === 1 ? ink.fg : ink.muted
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(X(p.img[0]), Y(p.img[1]), r, 0, Math.PI * 2)
    ctx.stroke()
    if (p.order === 1) {
      ctx.fillStyle = ink.muted
      ctx.fillText(p.label, X(p.img[0]) + 9, Y(p.img[1]) + 5)
    }
  }

  // estimated walls (last step)
  if (step.kind === "walls") {
    for (const l of final.lines) {
      const color = l.wall === null ? DATA.miss : l.wall === "south" || l.wall === "dup" ? ink.muted : DATA.hit
      drawLine(ctx, l.th, l.rho, sc.ap, bx, by, X, Y, color, l.wall === "south" || l.wall === "dup")
    }
  }

  // found paths as image points: ap + L * (s, sqrt(1 - s^2))
  const pts = foundAt(step)
  ctx.fillStyle = DATA.found
  for (const [s, L] of pts) {
    const c = Math.max(-1, Math.min(1, s))
    const px = sc.ap[0] + L * c
    const py = sc.ap[1] + L * Math.sqrt(1 - c * c)
    ctx.beginPath()
    ctx.arc(X(px), Y(py), 4, 0, Math.PI * 2)
    ctx.fill()
  }

  // AP and client
  ctx.fillStyle = ink.fg
  ctx.fillRect(X(sc.ap[0]) - 6, Y(sc.ap[1]) - 6, 12, 12)
  ctx.beginPath()
  ctx.arc(X(sc.client[0]), Y(sc.client[1]), 6, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = ink.muted
  ctx.fillText("AP", X(sc.ap[0]) + 10, Y(sc.ap[1]) + 18)
  ctx.fillText("用戶", X(sc.client[0]) + 10, Y(sc.client[1]) + 5)
  ctx.restore()
}

function drawLine(
  ctx: CanvasRenderingContext2D, th: number, rho: number, ap: [number, number],
  bx: [number, number], by: [number, number], X: (x: number) => number, Y: (y: number) => number,
  color: string, dashed: boolean,
) {
  const n = [Math.cos(th), Math.sin(th)]
  const foot = [ap[0] + rho * n[0], ap[1] + rho * n[1]]
  const t = [-n[1], n[0]]
  const len = (bx[1] - bx[0]) + (by[1] - by[0])
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.setLineDash(dashed ? [6, 5] : [])
  ctx.beginPath()
  ctx.moveTo(X(foot[0] - t[0] * len), Y(foot[1] - t[1] * len))
  ctx.lineTo(X(foot[0] + t[0] * len), Y(foot[1] + t[1] * len))
  ctx.stroke()
  ctx.setLineDash([])
}

// ------------------------------------------------------------ angle-delay map

const VIRIDIS = [
  [68, 1, 84], [65, 68, 135], [42, 120, 142], [34, 168, 132], [122, 209, 81], [253, 231, 37],
]

function cmap(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, v)) * (VIRIDIS.length - 1)
  const i = Math.min(VIRIDIS.length - 2, Math.floor(t))
  const f = t - i
  const a = VIRIDIS[i]
  const b = VIRIDIS[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

function mapImage(m: MapData, gate: number | undefined): HTMLCanvasElement {
  m.bytes ??= bytes(m.data)
  const c = document.createElement("canvas")
  c.width = m.ns
  c.height = m.nx
  const ctx = c.getContext("2d")!
  const img = ctx.createImageData(m.ns, m.nx)
  for (let i = 0; i < m.ns; i++) {
    for (let j = 0; j < m.nx; j++) {
      const v = m.bytes[i * m.nx + j]
      const o = ((m.nx - 1 - j) * m.ns + i) * 4
      let [r, g, b] = cmap(v / 255)
      if (gate !== undefined && v < gate) {
        const k = (r + g + b) / 3 * 0.35
        r = g = b = k
      }
      img.data[o] = r
      img.data[o + 1] = g
      img.data[o + 2] = b
      img.data[o + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return c
}

const cache = new WeakMap<MapData, HTMLCanvasElement>()

export function drawMap(canvas: HTMLCanvasElement, sc: Scene, step: Step, ink: Ink) {
  const { ctx, w, h } = setup(canvas)
  const pad = { l: 44, r: 8, t: 8, b: 40 }
  const pw = w - pad.l - pad.r
  const ph = h - pad.t - pad.b
  const lmax = sc.lmax
  const X = (s: number) => pad.l + ((s + 1) / 2) * pw
  const Y = (L: number) => pad.t + ph - (L / lmax) * ph

  if (step.kind !== "eig") {
    ctx.fillStyle = "#16161a"
    ctx.fillRect(pad.l, pad.t, pw, ph)
  }

  if (step.kind === "eig") {
    drawEig(ctx, step, pad, pw, ph, ink)
    return
  }

  if (step.map) {
    let img = cache.get(step.map)
    if (!img) {
      img = mapImage(step.map, step.gate)
      cache.set(step.map, img)
    }
    const m = step.map
    const ds = (m.s1 - m.s0) / Math.max(1, m.ns - 1)
    const dx = (m.x1 - m.x0) / Math.max(1, m.nx - 1)
    ctx.imageSmoothingEnabled = false
    ctx.save()
    ctx.beginPath()
    ctx.rect(pad.l, pad.t, pw, ph)
    ctx.clip()
    ctx.drawImage(img, X(m.s0 - ds / 2), Y(m.x1 + dx / 2), X(m.s1 + ds / 2) - X(m.s0 - ds / 2), Y(m.x0 - dx / 2) - Y(m.x1 + dx / 2))
    ctx.restore()
  }

  ctx.save()
  ctx.beginPath()
  ctx.rect(pad.l, pad.t, pw, ph)
  ctx.clip()

  if (step.lines) {
    ctx.strokeStyle = DATA.found
    ctx.lineWidth = 1.5
    for (const L of step.lines) {
      ctx.beginPath()
      ctx.moveTo(pad.l, Y(L))
      ctx.lineTo(pad.l + pw, Y(L))
      ctx.stroke()
    }
  }
  if (step.beams) {
    const b = step.beams
    b.bytes ??= bytes(b.data)
    const band = Math.max(3, ph / 80)
    const cw = pw / b.ns + 0.5
    for (let q = 0; q < b.k; q++) {
      const y = Y(step.lines![q])
      for (let i = 0; i < b.ns; i++) {
        const v = b.bytes[i * b.k + q] / 255
        const s = b.s0 + (i / (b.ns - 1)) * (b.s1 - b.s0)
        const [r, g, bl] = cmap(v)
        ctx.fillStyle = `rgb(${r},${g},${bl})`
        ctx.fillRect(X(s) - cw / 2, y - band, cw, band * 2)
      }
    }
  }

  // truth
  ctx.font = FONT
  for (const p of sc.paths) {
    ctx.strokeStyle = p.order <= 1 ? "#ffffff" : "rgba(255,255,255,0.55)"
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(X(p.s), Y(p.L), p.order <= 1 ? 7 : 5, 0, Math.PI * 2)
    ctx.stroke()
    if (p.order <= 1 && p.label !== "S") {
      ctx.fillStyle = "#ffffff"
      ctx.fillText(p.label, X(p.s) + 10, Y(p.L) - 6)
    }
  }

  // found and picked
  const cross = (pt: Pt, color: string, size = 6) => {
    ctx.strokeStyle = color
    ctx.lineWidth = 2.5
    ctx.beginPath()
    ctx.moveTo(X(pt[0]) - size, Y(pt[1]) - size)
    ctx.lineTo(X(pt[0]) + size, Y(pt[1]) + size)
    ctx.moveTo(X(pt[0]) + size, Y(pt[1]) - size)
    ctx.lineTo(X(pt[0]) - size, Y(pt[1]) + size)
    ctx.stroke()
  }
  for (const pt of step.found ?? []) cross(pt, DATA.found, 5)
  for (const pt of step.picked ?? []) cross(pt, DATA.found)
  for (const pt of step.dropped ?? []) cross(pt, DATA.miss)
  if (step.grid) {
    ctx.strokeStyle = DATA.found
    ctx.lineWidth = 2
    ctx.strokeRect(X(step.grid[0]) - 7, Y(step.grid[1]) - 7, 14, 14)
    if (step.refined) {
      ctx.beginPath()
      ctx.moveTo(X(step.grid[0]), Y(step.grid[1]))
      ctx.lineTo(X(step.refined[0]), Y(step.refined[1]))
      ctx.stroke()
      cross(step.refined, DATA.found, 7)
    }
  }
  ctx.restore()

  // axes
  ctx.fillStyle = ink.muted
  ctx.strokeStyle = ink.faint
  ctx.font = FONT
  ctx.textAlign = "center"
  for (const deg of [180, 135, 90, 45, 0]) {
    const x = X(Math.cos((deg * Math.PI) / 180))
    ctx.fillText(`${deg}°`, x, pad.t + ph + 18)
  }
  ctx.fillText("到達角", pad.l + pw / 2, pad.t + ph + 36)
  ctx.textAlign = "right"
  const stepL = lmax > 12 ? 5 : 2
  for (let L = 0; L <= lmax; L += stepL) ctx.fillText(`${L}`, pad.l - 8, Y(L) + 5)
  ctx.save()
  ctx.translate(12, pad.t + ph / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = "center"
  ctx.fillText("路徑長 (m)", 0, 0)
  ctx.restore()
  ctx.textAlign = "left"
}

function drawEig(ctx: CanvasRenderingContext2D, step: Step, pad: { l: number; t: number }, pw: number, ph: number, ink: Ink) {
  const ev = step.ev ?? []
  const lo = Math.min(...ev, step.thr ?? 0) - 5
  const hi = Math.max(...ev) + 3
  const Y = (v: number) => pad.t + ph - ((v - lo) / (hi - lo)) * ph
  const bw = pw / ev.length
  ev.forEach((v, i) => {
    ctx.fillStyle = i < (step.k ?? 0) ? DATA.found : ink.muted
    ctx.fillRect(pad.l + i * bw + 1, Y(v), Math.max(1, bw - 2), pad.t + ph - Y(v))
  })
  if (step.thr !== undefined) {
    ctx.strokeStyle = ink.fg
    ctx.setLineDash([6, 5])
    ctx.beginPath()
    ctx.moveTo(pad.l, Y(step.thr))
    ctx.lineTo(pad.l + pw, Y(step.thr))
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = ink.fg
    ctx.font = FONT
    ctx.fillText("雜訊門檻", pad.l + pw - 70, Y(step.thr) - 8)
  }
  ctx.fillStyle = ink.muted
  ctx.font = FONT
  ctx.textAlign = "center"
  ctx.fillText("特徵值(由大到小)", pad.l + pw / 2, pad.t + ph + 28)
  ctx.textAlign = "left"
}
