import { useEffect, useMemo, useRef, useState } from "react"
import { TaskShell } from "@/components/task-shell"
import { TaskThemeToggle } from "@/components/task-theme"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import config from "@/task.config.json"
import { DATA, drawMap, drawRoom, readInk } from "./draw"
import { METHODS, foundAt, wallSummary, type Final, type Step, type Trace } from "./trace"

const WALL_NAME: Record<string, string> = { east: "東", north: "北", west: "西" }

function caption(method: string, step: Step, index: number, steps: Step[], final: Final): string {
  const n = steps.slice(0, index + 1).filter(s => s.kind === "pick").length
  switch (step.kind) {
    case "map":
      return "2D FFT：把每支天線、每個子載波的 CSI 轉成到達角與路徑長的地圖。"
    case "peaks":
      return method === "music"
        ? `取 ${step.picked!.length + (step.dropped?.length ?? 0)} 個峰，強度太弱的丟掉，留下 ${step.picked!.length} 條路徑。`
        : `取局部最大值，超過門檻的算 1 條路徑，共 ${step.picked!.length} 條。灰色區域低於門檻。`
    case "pick":
      return method === "nomp"
        ? `第 ${n} 條：在殘差裡取最強的格點(方框)，再用牛頓法移到格點之間(叉)，然後從 CSI 扣掉。`
        : `第 ${n} 條：在殘差裡取最強的格點，從 CSI 扣掉，再找下一條。`
    case "stop":
      return `殘差全部低於門檻，停止。共找到 ${step.found!.length} 條路徑。`
    case "eig":
      return `平滑後的共變異矩陣：有 ${step.k} 個特徵值高於雜訊門檻，所以要找 ${step.k} 條路徑。`
    case "spectrum":
      return "MUSIC 頻譜：越接近訊號子空間的位置越亮，峰比 FFT 尖。"
    case "delays":
      return `只看延遲：解出 ${step.lines!.length} 條路徑長，還不知道角度。`
    case "beams":
      return "每條路徑長各做一次波束掃描，取最亮的角度。紅叉太弱，丟掉。"
    case "walls": {
      const { found, falses, hits } = wallSummary(final)
      const err = hits.map(h => `${WALL_NAME[h.wall!]} ${h.err?.toFixed(2)} m`).join("、")
      return `每條反射換成 1 面牆(用戶與鏡像點的中垂線)。找到 ${found} / 3 面${err ? `(${err})` : ""}，假牆 ${falses} 面。`
    }
  }
}

export default function App() {
  const lang = "zh-TW" as const
  const [trace, setTrace] = useState<Trace | null>(null)
  const [scene, setScene] = useState(0)
  const [bw, setBw] = useState("160")
  const [method, setMethod] = useState<string>("music")
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [themeTick, setThemeTick] = useState(0)
  const roomRef = useRef<HTMLCanvasElement>(null)
  const mapRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    fetch("./wall-mapping-assets/trace.json").then(r => r.json()).then(setTrace)
  }, [])

  useEffect(() => {
    const obs = new MutationObserver(() => setThemeTick(t => t + 1))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    const onResize = () => setThemeTick(t => t + 1)
    window.addEventListener("resize", onResize)
    return () => {
      obs.disconnect()
      window.removeEventListener("resize", onResize)
    }
  }, [])

  const sc = trace?.scenes[scene]
  const run = sc?.runs[bw]?.[method]
  const steps = useMemo<Step[]>(() => {
    if (!run) return []
    const last = run.steps[run.steps.length - 1]
    return [...run.steps, { kind: "walls", picked: last ? foundAt(last) : [] }]
  }, [run])
  const step = steps[Math.min(index, steps.length - 1)]

  const restart = () => {
    setIndex(0)
    setPlaying(true)
  }

  const last = steps.length - 1
  const running = playing && index < last
  useEffect(() => {
    if (!running) return
    const t = setTimeout(() => {
      setIndex(i => i + 1)
      if (index + 1 >= last) setPlaying(false)
    }, 1100)
    return () => clearTimeout(t)
  }, [running, index, last])

  useEffect(() => {
    if (!sc || !run || !step || !roomRef.current || !mapRef.current) return
    const ink = readInk()
    drawRoom(roomRef.current, sc, step, run.final, ink)
    drawMap(mapRef.current, sc, step, ink)
  }, [sc, run, step, themeTick])

  return (
    <TaskShell title={config.title} description={config.description} lang={lang} wide actions={<TaskThemeToggle lang={lang} />}>
      {!trace || !sc || !run || !step ? (
        <p className="text-body text-muted-foreground">載入中</p>
      ) : (
        <section aria-label="工作區" className="flex flex-col gap-6">
          <div className="flex flex-wrap items-center gap-x-10 gap-y-4">
            <Tabs value={method} onValueChange={v => { setMethod(String(v)); restart() }}>
              <TabsList aria-label="方法" className="h-auto flex-wrap">
                {METHODS.map(m => <TabsTrigger key={m.id} value={m.id}>{m.name}</TabsTrigger>)}
              </TabsList>
            </Tabs>
            <Tabs value={bw} onValueChange={v => { setBw(String(v)); restart() }}>
              <TabsList aria-label="頻寬">
                {trace.bws.map(b => <TabsTrigger key={b} value={String(b)}>{b} MHz</TabsTrigger>)}
              </TabsList>
            </Tabs>
            <div className="flex items-center gap-2">
              <span className="text-body text-muted-foreground">房間</span>
              <Tabs value={String(scene)} onValueChange={v => { setScene(Number(v)); restart() }}>
                <TabsList aria-label="房間">
                  {trace.scenes.map((_, i) => <TabsTrigger key={i} value={String(i)}>{i + 1}</TabsTrigger>)}
                </TabsList>
              </Tabs>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <Button variant="outline" onClick={() => {
              if (running) return setPlaying(false)
              if (index >= last) setIndex(0)
              setPlaying(true)
            }}>{running ? "暫停" : "播放"}</Button>
            <Button variant="ghost" disabled={index === 0} onClick={() => { setPlaying(false); setIndex(i => Math.max(0, i - 1)) }}>上一步</Button>
            <Button variant="ghost" disabled={index >= steps.length - 1} onClick={() => { setPlaying(false); setIndex(i => Math.min(steps.length - 1, i + 1)) }}>下一步</Button>
            <Slider
              aria-label="步驟"
              className="min-w-40 flex-1"
              min={0}
              max={steps.length - 1}
              value={[index]}
              onValueChange={v => { setPlaying(false); setIndex(Array.isArray(v) ? v[0] : v) }}
            />
            <span className="text-caption text-muted-foreground tabular-nums">{index + 1} / {steps.length}</span>
          </div>

          <p className="min-h-12 text-body" aria-live="polite">{caption(method, step, index, steps, run.final)}</p>

          <div className="grid gap-8 md:grid-cols-2">
            <figure className="flex flex-col gap-2">
              <figcaption className="text-caption text-muted-foreground">房間(俯視)</figcaption>
              <canvas ref={roomRef} className="aspect-[6/5] w-full" aria-label="房間俯視圖" />
            </figure>
            <figure className="flex flex-col gap-2">
              <figcaption className="text-caption text-muted-foreground">{step.kind === "eig" ? "特徵值" : "到達角與路徑長"}</figcaption>
              <canvas ref={mapRef} className="aspect-[6/5] w-full" aria-label="到達角與路徑長地圖" />
            </figure>
          </div>

          <ul className="flex flex-wrap gap-x-6 gap-y-2 text-caption text-muted-foreground">
            <li className="flex items-center gap-2"><span className="inline-block size-3 rounded-full border-2 border-foreground" />真實鏡像點</li>
            <li className="flex items-center gap-2"><span className="inline-block size-3 rounded-full" style={{ background: DATA.found }} />找到的路徑</li>
            <li className="flex items-center gap-2"><span className="inline-block h-0.5 w-4" style={{ background: DATA.hit }} />找對的牆</li>
            <li className="flex items-center gap-2"><span className="inline-block h-0.5 w-4" style={{ background: DATA.miss }} />假牆</li>
          </ul>
        </section>
      )}
    </TaskShell>
  )
}
