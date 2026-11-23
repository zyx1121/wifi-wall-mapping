import { useEffect, useRef, useState } from "react"
import { TaskShell } from "@/components/task-shell"
import { TaskThemeToggle } from "@/components/task-theme"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import config from "@/task.config.json"
import { METHODS, drawFrame, score, type Walk } from "./walk"

const SPEED = 6 // positions per second (0.3 m apart)

export default function App() {
  const lang = "zh-TW" as const
  const [walk, setWalk] = useState<Walk | null>(null)
  const [method, setMethod] = useState("music")
  const [bw, setBw] = useState("160")
  const [playing, setPlaying] = useState(true)
  const [n, setN] = useState(1)
  const canvas = useRef<HTMLCanvasElement>(null)
  const t = useRef(0)

  useEffect(() => {
    fetch("./wall-mapping-assets/walk.json").then(r => r.json()).then(setWalk)
  }, [])

  const run = walk?.runs[method]?.[bw]
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
      if (canvas.current) drawFrame(canvas.current, walk, run, t.current)
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [walk, run, playing, total])

  const s = run && walk ? score(run.fused[n - 1] ?? [], walk.targets) : null

  return (
    <TaskShell title={config.title} lang={lang} full actions={<TaskThemeToggle lang={lang} />}>
      {!walk || !run || !s ? (
        <p className="text-body text-muted-foreground">載入中</p>
      ) : (
        <section aria-label="工作區" className="flex min-h-0 flex-1 flex-col gap-4">
          <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
            <Tabs value={method} onValueChange={v => setMethod(String(v))}>
              <TabsList aria-label="方法" className="h-auto flex-wrap">
                {METHODS.map(m => <TabsTrigger key={m.id} value={m.id}>{m.name}</TabsTrigger>)}
              </TabsList>
            </Tabs>
            <Tabs value={bw} onValueChange={v => setBw(String(v))}>
              <TabsList aria-label="頻寬">
                {walk.bws.map(b => <TabsTrigger key={b} value={String(b)}>{b} MHz</TabsTrigger>)}
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
            <span className="flex items-center gap-2"><span className="inline-block h-1 w-4 bg-[#2fbf71]" />找到 {s.found} / {walk.targets.length} 面牆</span>
            <span className="flex items-center gap-2"><span className="inline-block h-1 w-4 bg-[#ef4f4f]" />假牆 {s.falses} 面</span>
          </p>
        </section>
      )}
    </TaskShell>
  )
}
