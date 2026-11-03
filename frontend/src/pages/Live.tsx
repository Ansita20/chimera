import { useEffect, useRef, useState } from 'react'
import { LineChart, Legend } from '../components/Charts'
import {
  ErrorLine,
  ExplanationCard,
  JobProgress,
  ModelScores,
  RegionBoxes,
  RegionLabels,
  Tile,
  VerdictBadge,
  modelColor,
  modelDash,
  useJob,
} from '../components/common'
import { api, MODEL_LABEL, MODEL_NAMES, wsUrl, type FrameResult, type Scene } from '../lib/api'

type Mode = 'idle' | 'detect' | 'capture'

/** One reply per message: requests resolve in order. */
class StreamClient {
  private waiters: ((m: any) => void)[] = []
  private constructor(private ws: WebSocket) {
    ws.binaryType = 'arraybuffer'
    ws.onmessage = (e) => this.waiters.shift()?.(JSON.parse(e.data))
    ws.onclose = () => this.waiters.splice(0).forEach((w) => w({ type: 'error', message: 'connection closed' }))
  }
  static open(url: string): Promise<StreamClient> {
    return new Promise((res, rej) => {
      const ws = new WebSocket(url)
      ws.onopen = () => res(new StreamClient(ws))
      ws.onerror = () => rej(new Error('could not connect to the CHIMERA stream endpoint'))
    })
  }
  request(data: string | ArrayBuffer): Promise<any> {
    return new Promise((res) => {
      this.waiters.push(res)
      this.ws.send(data)
    })
  }
  cmd(obj: object) {
    return this.request(JSON.stringify(obj))
  }
  close() {
    this.ws.close()
  }
}

interface Hist {
  i: number
  score: number
  unc: number
  m: Record<string, number>
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export default function Live({ scene, onSceneChanged }: { scene: Scene | null; onSceneChanged: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'))
  const clientRef = useRef<StreamClient | null>(null)
  const runRef = useRef(false)
  const [camOn, setCamOn] = useState(false)
  const [mode, setMode] = useState<Mode>('idle')
  const [error, setError] = useState<string | null>(null)
  const [res, setRes] = useState<FrameResult | null>(null)
  const [hist, setHist] = useState<Hist[]>([])
  const [feed, setFeed] = useState<{ id: number; track_id: number; summary: string; peak: number; at: string }[]>([])
  const [fps, setFps] = useState(6)
  const [width, setWidth] = useState(320)
  const [threshold, setThreshold] = useState(1)
  const [fusion, setFusion] = useState('max')
  const [record, setRecord] = useState(true)
  const [captureSecs, setCaptureSecs] = useState(15)
  const [captured, setCaptured] = useState(0)
  const [measuredFps, setMeasuredFps] = useState(0)
  const [jobId, setJobId] = useState<number | null>(null)
  const [sessionId, setSessionId] = useState<number | null>(null)
  const trained = scene ? MODEL_NAMES.filter((m) => scene.models[m]?.trained) : []
  const [models, setModels] = useState<string[]>([])
  useEffect(() => setModels(trained), [scene?.id, trained.join()])
  const job = useJob(jobId, () => onSceneChanged())

  useEffect(() => () => void stopAll(true), []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    stopAll(true)
    setRes(null)
    setHist([])
    setFeed([])
  }, [scene?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  async function startCamera() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
      const v = videoRef.current!
      v.srcObject = stream
      await v.play()
      setCamOn(true)
    } catch (e) {
      setError(`Camera unavailable: ${(e as Error).message}. Browsers only allow camera access on https or localhost.`)
    }
  }

  function stopCamera() {
    const v = videoRef.current
    ;(v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop())
    if (v) v.srcObject = null
    setCamOn(false)
  }

  async function grab(): Promise<ArrayBuffer | null> {
    const v = videoRef.current
    if (!v || !v.videoWidth) return null
    const c = canvasRef.current
    c.width = width
    c.height = Math.round((width * v.videoHeight) / v.videoWidth)
    c.getContext('2d')!.drawImage(v, 0, 0, c.width, c.height)
    const blob = await new Promise<Blob | null>((r) => c.toBlob(r, 'image/jpeg', 0.85))
    return blob ? blob.arrayBuffer() : null
  }

  async function connect(): Promise<StreamClient> {
    if (!scene) throw new Error('select a scene first')
    if (!clientRef.current) clientRef.current = await StreamClient.open(wsUrl(`/api/scenes/${scene.id}/stream`))
    return clientRef.current
  }

  async function loop(onFrame: (m: any) => boolean) {
    runRef.current = true
    const times: number[] = []
    while (runRef.current) {
      const t0 = performance.now()
      const buf = await grab()
      if (!buf) {
        await sleep(100)
        continue
      }
      const c = clientRef.current
      if (!c) break
      const m = await c.request(buf)
      if (m.type === 'error') {
        setError(m.message)
        if (m.message === 'connection closed') break
      } else if (!onFrame(m)) break
      times.push(performance.now())
      if (times.length > 10) times.shift()
      if (times.length > 1) setMeasuredFps(((times.length - 1) * 1000) / (times[times.length - 1] - times[0]))
      await sleep(Math.max(0, 1000 / fps - (performance.now() - t0)))
    }
    runRef.current = false
  }

  async function startDetect() {
    setError(null)
    try {
      if (!camOn) await startCamera()
      const c = await connect()
      const r = await c.cmd({ type: 'start', models, threshold, fusion, record, source: 'webcam' })
      if (r.type === 'error') throw new Error(r.message)
      setSessionId(r.session_id)
      setMode('detect')
      setHist([])
      loop((m) => {
        if (m.type !== 'result') return true
        const fr = m as FrameResult
        setRes(fr)
        setHist((h) => [
          ...h.slice(-239),
          {
            i: fr.frame_idx,
            score: fr.score,
            unc: fr.uncertainty,
            m: Object.fromEntries(Object.entries(fr.models).map(([k, v]) => [k, v.score / threshold])),
          },
        ])
        if (fr.events_new?.length)
          setFeed((f) => [...fr.events_new!.map((e) => ({ ...e, at: new Date().toLocaleTimeString() })), ...f].slice(0, 30))
        return true
      })
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function startCapture() {
    setError(null)
    try {
      if (!camOn) await startCamera()
      const c = await connect()
      await c.cmd({ type: 'capture' })
      setMode('capture')
      setCaptured(0)
      const target = Math.round(captureSecs * fps)
      await loop((m) => {
        if (m.type === 'captured') setCaptured(m.count)
        return m.count < target
      })
      const r = await c.cmd({ type: 'stop' })
      setMode('idle')
      onSceneChanged()
      if (r.captured > 0 && scene) {
        const { job_id } = await api.train(scene.id, ['embedding', 'autoencoder', 'temporal'])
        setJobId(job_id)
      }
    } catch (e) {
      setError((e as Error).message)
      setMode('idle')
    }
  }

  async function stopAll(silent = false) {
    runRef.current = false
    const c = clientRef.current
    if (c && !silent) {
      await sleep(50)
      try {
        await c.cmd({ type: 'stop' })
      } catch {
        /* ignore */
      }
    }
    if (silent && c) {
      c.close()
      clientRef.current = null
    }
    setMode('idle')
  }

  async function changeThreshold(v: number) {
    setThreshold(v)
    if (mode === 'detect' && clientRef.current && !runRef.current) await clientRef.current.cmd({ type: 'config', threshold: v })
  }

  const x = hist.map((h) => h.i)
  const lastModels = res ? Object.keys(res.models) : []
  const training = job && (job.status === 'running' || job.status === 'queued')

  return (
    <div className="live-grid">
      <div className="stack">
        <div className="viewer">
          <video ref={videoRef} muted playsInline style={{ display: camOn ? 'block' : 'none' }} />
          {!camOn && (
            <div className="placeholder">
              <div>
                <div style={{ fontSize: 16, color: '#555' }}>Camera is off</div>
                <div className="small">Start the camera, then teach CHIMERA what normal looks like.</div>
              </div>
            </div>
          )}
          {camOn && mode === 'detect' && res && <img className="heat" src={res.heatmap} alt="" />}
          {camOn && mode === 'detect' && res && <RegionBoxes regions={res.regions} />}
          {camOn && mode === 'detect' && res && <RegionLabels regions={res.regions} />}
          {mode !== 'idle' && (
            <div className="rec">
              {mode === 'capture' ? `● learning normality ${captured}/${Math.round(captureSecs * fps)}` : `● detecting · ${measuredFps.toFixed(1)} fps`}
            </div>
          )}
        </div>

        <div className="card stack">
          <div className="row">
            {!camOn ? (
              <button className="btn" onClick={startCamera}>Start camera</button>
            ) : (
              <button className="btn" onClick={() => (stopAll(true), stopCamera())}>Stop camera</button>
            )}
            <button className="btn" disabled={!scene || mode !== 'idle' || !!training} onClick={startCapture}
              title="Record normal footage, then train all detectors on it">
              Learn normality ({captureSecs}s)
            </button>
            {mode === 'detect' ? (
              <button className="btn danger" onClick={() => stopAll()}>Stop detection</button>
            ) : (
              <button className="btn primary" disabled={!scene || !models.length || mode !== 'idle'} onClick={startDetect}>
                Detect anomalies
              </button>
            )}
            {mode === 'capture' && <button className="btn danger" onClick={() => (runRef.current = false)}>Finish capture</button>}
          </div>
          {!scene && <div className="muted">Create or select a scene in the top bar first.</div>}
          {scene && !trained.length && !training && (
            <div className="alert">
              <span className="icon">i</span>
              <span>
                No trained detectors yet. Point the camera at the scene in its normal state and press <b>Learn normality</b> —
                CHIMERA records {captureSecs}s of footage and trains all three detectors on it. Keep unusual objects out of view while it learns.
              </span>
            </div>
          )}
          {job && <JobProgress job={job} />}
          <div className="row" style={{ gap: 16 }}>
            <label className="field">
              Threshold · {threshold.toFixed(2)}× calibrated
              <input type="range" min={0.5} max={3} step={0.05} value={threshold} disabled={mode === 'detect'}
                onChange={(e) => changeThreshold(Number(e.target.value))} />
            </label>
            <label className="field">
              Frame rate
              <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
                {[2, 4, 6, 8, 10, 15].map((f) => <option key={f} value={f}>{f} fps</option>)}
              </select>
            </label>
            <label className="field">
              Resolution
              <select value={width} onChange={(e) => setWidth(Number(e.target.value))}>
                {[240, 320, 480, 640].map((w) => <option key={w} value={w}>{w}px</option>)}
              </select>
            </label>
            <label className="field">
              Fusion
              <select value={fusion} disabled={mode === 'detect'} onChange={(e) => setFusion(e.target.value)}>
                <option value="max">any detector (max)</option>
                <option value="mean">consensus (mean)</option>
              </select>
            </label>
            <label className="field">
              Learn for
              <select value={captureSecs} onChange={(e) => setCaptureSecs(Number(e.target.value))}>
                {[8, 15, 30, 60].map((s) => <option key={s} value={s}>{s}s</option>)}
              </select>
            </label>
          </div>
          <div className="row" style={{ gap: 16 }}>
            {MODEL_NAMES.map((m) => (
              <label key={m} className="row small" style={{ gap: 6, opacity: trained.includes(m) ? 1 : 0.5 }}>
                <input type="checkbox" disabled={!trained.includes(m) || mode === 'detect'} checked={models.includes(m)}
                  onChange={(e) => setModels((s) => (e.target.checked ? [...s, m] : s.filter((x) => x !== m)))} />
                <span className="model-dot" style={{ background: modelColor(m) }} />
                {MODEL_LABEL[m]} {!trained.includes(m) && '(untrained)'}
              </label>
            ))}
            <label className="row small" style={{ gap: 6 }}>
              <input type="checkbox" checked={record} disabled={mode === 'detect'} onChange={(e) => setRecord(e.target.checked)} />
              Record session & events
            </label>
          </div>
          <ErrorLine error={error} />
        </div>

        <div className="card">
          <div className="between" style={{ marginBottom: 8 }}>
            <h3>Anomaly score over time</h3>
            <Legend items={[{ label: 'Fused', color: 'var(--text-primary)', line: true }, ...lastModels.map((m) => ({ label: MODEL_LABEL[m], color: modelColor(m), dash: modelDash(m), line: true }))]} />
          </div>
          {hist.length ? (
            <LineChart
              x={x}
              height={200}
              refLine={{ y: 1, label: 'threshold' }}
              xFormat={(v) => `frame ${Math.round(v)}`}
              series={[
                { key: 'fused', label: 'Fused', color: 'var(--text-primary)', values: hist.map((h) => h.score) },
                ...lastModels.map((m) => ({ key: m, label: MODEL_LABEL[m], color: modelColor(m), dash: modelDash(m), values: hist.map((h) => h.m[m] ?? null) })),
              ]}
              yFormat={(v) => `${v.toFixed(1)}×`}
            />
          ) : (
            <div className="empty">Start detection to see the live score.</div>
          )}
          {hist.length > 0 && (
            <>
              <h3 style={{ margin: '12px 0 4px' }}>Uncertainty</h3>
              <LineChart x={x} height={110} yMax={1} xFormat={(v) => `frame ${Math.round(v)}`} yFormat={(v) => `${Math.round(v * 100)}%`}
                series={[{ key: 'u', label: 'Uncertainty', color: 'var(--text-secondary)', values: hist.map((h) => h.unc) }]} />
            </>
          )}
        </div>
      </div>

      <div className="stack">
        <div className="card stack">
          <div className="between">
            <VerdictBadge verdict={mode === 'idle' && !res ? 'idle' : res?.verdict ?? 'warming_up'} big />
            {sessionId && <span className="small muted">session #{sessionId}</span>}
          </div>
          <div className="tiles">
            <Tile label="Score" value={res ? `${res.score.toFixed(2)}×` : '–'} sub="1.0× = threshold" />
            <Tile label="Uncertainty" value={res ? `${Math.round(res.uncertainty * 100)}%` : '–'} />
            <Tile label="Latency" value={res ? `${Math.round(res.latency_ms)}` : '–'} sub="ms / frame" />
          </div>
          {res?.drift.alert && (
            <div className="alert">
              <span className="icon" style={{ color: 'var(--warning)' }}>⚠</span>
              <span>
                The scene has looked abnormal for a while (drift {res.drift.ema.toFixed(2)}×). If this is the new normal
                (moved camera, new lighting), press <b>Learn normality</b> again.
              </span>
            </div>
          )}
          <ModelScores models={res?.models ?? {}} threshold={threshold} />
        </div>

        <div className="card stack">
          <h3>Flagged regions {res?.regions.length ? `(${res.regions.length})` : ''}</h3>
          {res?.regions.length ? (
            res.regions.slice(0, 4).map((r, i) => (
              <ExplanationCard key={r.track_id ?? i} e={r.explanation}
                title={`${r.track_id ? `#${r.track_id} · ` : ''}${r.peak.toFixed(1)}× · ${r.age_s.toFixed(1)}s${r.confirmed ? '' : ' (tentative)'}`} />
            ))
          ) : (
            <div className="muted small">Nothing unusual in view.</div>
          )}
        </div>

        <div className="card stack">
          <h3>Event feed</h3>
          {feed.length ? (
            feed.map((e) => (
              <div key={e.id} className="between small">
                <span>
                  <b>#{e.track_id}</b> {e.summary}
                </span>
                <span className="muted mono">{e.peak.toFixed(1)}× · {e.at}</span>
              </div>
            ))
          ) : (
            <div className="muted small">Regions that persist for {3}+ frames become events and are saved with a snapshot.</div>
          )}
        </div>
      </div>
    </div>
  )
}
