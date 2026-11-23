import { useEffect, useRef, useState } from "react"
import { TaskShell } from "@/components/task-shell"
import { TaskThemeToggle } from "@/components/task-theme"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import config from "@/task.config.json"
import { HEAT_MAX, HEAT_MIN, drawFrame, heatColor, score, snapAt, type Layer, type Walk } from "./walk"

const LAYERS: { id: Layer; name: string }[] = [
  { id: "walls", name: "牆面" },
  { id: "rooms", name: "房間" },
  { id: "heat", name: "訊號" },
]

const GRADIENT = `linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1].map(f => `rgb(${heatColor(HEAT_MIN + f * (HEAT_MAX - HEAT_MIN)).join(",")})`).join(", ")})`

const SPEED = 6 // positions per second (0.3 m apart)

export default function App() {
  const lang = "zh-TW" as const
  const [walk, setWalk] = useState<Walk | null>(null)
  const [layer, setLayer] = useState<Layer>("walls")
  const [playing, setPlaying] = useState(true)
  const [n, setN] = useState(1)
  const canvas = useRef<HTMLCanvasElement>(null)
  const t = useRef(0)

  useEffect(() => {
    fetch("./wall-mapping-assets/walk.json").then(r => r.json()).then(setWalk)
  }, [])

  const run = walk?.run
  const total = walk?.positions.length ?? 1

  // one render loop: advances the playhead while playing, redraws every frame
  useEffect(() => {
    if (!walk || !run) return
    let raf = 0
    let last = performance.now()
    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      if (playing) {
        t.current = Math.min(total - 1, t.current + dt * SPEED)
        if (t.current >= total - 1) setPlaying(false)
      }
      const next = Math.floor(t.current) + 1
      setN(prev => (prev === next ? prev : next))
      if (canvas.current) drawFrame(canvas.current, walk, run, t.current, layer)
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [walk, run, playing, total, layer])

  const s = run && walk ? score(run.fused[n - 1] ?? [], walk.targets) : null
  const snap = run ? snapAt(run, n) : null

  return (
    <TaskShell title={config.title} lang={lang} full actions={<TaskThemeToggle lang={lang} />}>
      {!walk || !run || !s ? (
        <p className="text-body text-muted-foreground">載入中</p>
      ) : (
        <section aria-label="工作區" className="flex min-h-0 flex-1 flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
            <Tabs value={layer} onValueChange={v => setLayer(v as Layer)}>
              <TabsList aria-label="圖層">
                {LAYERS.map(l => <TabsTrigger key={l.id} value={l.id}>{l.name}</TabsTrigger>)}
              </TabsList>
            </Tabs>
          </div>

          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={() => {
              if (playing) return setPlaying(false)
              if (t.current >= total - 1) t.current = 0
              setPlaying(true)
            }}>{playing ? "暫停" : "播放"}</Button>
            <Slider
              aria-label="走到第幾個位置"
              className="flex-1"
              min={1}
              max={total}
              value={[n]}
              onValueChange={v => {
                const pick = Array.isArray(v) ? v[0] : v
                if (!Number.isFinite(pick)) return
                setPlaying(false)
                t.current = Math.max(0, Math.min(total - 1, pick - 1))
              }}
            />
          </div>

          <canvas ref={canvas} className="min-h-0 w-full flex-1" aria-label="公寓平面圖" />

          <p className="flex flex-wrap items-center gap-x-6 gap-y-1 text-body tabular-nums">
            <span>第 {n} / {total} 個位置</span>
            {layer === "walls" && (
              <>
                <span className="flex items-center gap-2"><span className="inline-block h-1 w-4 bg-[#2fbf71]" />找到 {s.found} / {walk.targets.length} 面牆</span>
                <span className="flex items-center gap-2"><span className="inline-block h-1 w-4 bg-[#ef4f4f]" />假牆 {s.falses} 面</span>
              </>
            )}
            {layer === "rooms" && <span>分出 {snap?.rooms_ok ?? 0} / {walk.plan.rooms_true} 間房</span>}
            {layer === "heat" && (
              <>
                <span className="flex items-center gap-2">{HEAT_MIN}<span className="inline-block h-2 w-20" style={{ background: GRADIENT }} />{HEAT_MAX} dBm</span>
                {snap && <span>隔牆房間誤差 {snap.heat_err} dB</span>}
              </>
            )}
          </p>
        </section>
      )}
    </TaskShell>
  )
}
