// Types and decoding for public/trace.json, written by experiments/demo_trace.py.

export type Pt = [number, number] // [s = cos(phi), path length in m]

export type MapData = {
  ns: number
  nx: number
  s0: number
  s1: number
  x0: number
  x1: number
  data: string
  bytes?: Uint8Array
}

export type Step = {
  kind: "map" | "peaks" | "pick" | "stop" | "eig" | "spectrum" | "delays" | "beams" | "walls"
  map?: MapData
  gate?: number
  picked?: Pt[]
  found?: Pt[]
  dropped?: Pt[]
  grid?: Pt
  refined?: Pt
  ev?: number[]
  thr?: number
  k?: number
  lines?: number[]
  beams?: { ns: number; k: number; s0: number; s1: number; data: string; bytes?: Uint8Array }
}

export type WallLine = { th: number; rho: number; wall: string | null; err: number | null }

export type Final = { los: [number, number] | null; images: [number, number][]; lines: WallLine[] }

export type Run = { steps: Step[]; final: Final }

export type TruePath = { label: string; order: number; s: number; L: number; img: [number, number]; power: number }

export type Scene = {
  seed: number
  room: [number, number]
  ap: [number, number]
  client: [number, number]
  d0: number
  lmax: number
  walls: { name: string; p0: [number, number]; p1: [number, number] }[]
  paths: TruePath[]
  runs: Record<string, Record<string, Run>>
}

export type Trace = { bws: number[]; ants: number; snr: number; dbRange: number; scenes: Scene[] }

export const METHODS = [
  { id: "ifft", name: "IFFT" },
  { id: "omp", name: "OMP" },
  { id: "nomp", name: "NOMP" },
  { id: "music", name: "MUSIC" },
  { id: "mpencil", name: "Matrix Pencil" },
] as const

export function bytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

/** All points found so far, as path-space points. */
export function foundAt(step: Step): Pt[] {
  if (step.kind === "pick") return [...(step.found ?? []), ...(step.refined ? [step.refined] : step.grid ? [step.grid] : [])]
  return step.found ?? step.picked ?? []
}

const TARGETS = ["east", "north", "west"]

export function wallSummary(final: Final) {
  const hits = final.lines.filter(l => l.wall && TARGETS.includes(l.wall))
  const falses = final.lines.filter(l => l.wall === null).length
  return { found: hits.length, falses, hits }
}
