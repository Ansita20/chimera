import { useEffect, useState } from 'react'
import { ErrorLine, JobProgress, fmtTime, modelColor, useJob } from '../components/common'
import { api, MODEL_LABEL, MODEL_NAMES, type Job, type Scene } from '../lib/api'

export default function Setup({ scene, onSceneChanged, onOpenSession }: {
  scene: Scene | null
  onSceneChanged: () => void
  onOpenSession: (id: number) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [trainModels, setTrainModels] = useState<string[]>([...MODEL_NAMES])
  const [epochs, setEpochs] = useState<string>('')
  const [trainJob, setTrainJob] = useState<number | null>(null)
  const [videoJob, setVideoJob] = useState<number | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [vfps, setVfps] = useState(10)
  const tj = useJob(trainJob, () => (onSceneChanged(), refreshJobs()))
  const vj = useJob(videoJob, () => refreshJobs())

  const refreshJobs = () => scene && api.jobs(scene.id).then(setJobs).catch(() => {})
  useEffect(() => {
    refreshJobs()
  }, [scene?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!scene) return <div className="card empty">Create a scene in the top bar. A scene is one notion of “normal” — a camera view or a product type.</div>

  async function upload(files: FileList | null) {
    if (!files?.length || !scene) return
    setBusy(true)
    setError(null)
    try {
      const r = await api.uploadNormal(scene.id, Array.from(files))
      if (r.errors.length) setError(r.errors.join('; '))
      onSceneChanged()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  async function train() {
    if (!scene) return
    setError(null)
    try {
      setTrainJob((await api.train(scene.id, trainModels, epochs ? Number(epochs) : undefined)).job_id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function analyzeVideo(f: File | undefined) {
    if (!f || !scene) return
    setError(null)
    try {
      setVideoJob((await api.analyzeVideo(scene.id, f, { fps: vfps })).job_id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const n = scene.normal
  return (
    <div className="stack">
      <ErrorLine error={error} />
      <div className="grid cols-2">
        <div className="card stack">
          <h3>1 · Normal data</h3>
          <div className="tiles">
            <div className="tile"><div className="label">Still images</div><div className="value">{n.images}</div></div>
            <div className="tile"><div className="label">Sequences</div><div className="value">{n.sequences}</div><div className="sub">{n.sequence_frames} frames</div></div>
          </div>
          <p className="small secondary" style={{ margin: 0 }}>
            Upload images or videos of the scene in its <b>normal</b> state only, or capture from the webcam in <b>Live</b>.
            The appearance detectors learn from stills and frames; the temporal detector needs contiguous video sequences.
          </p>
          <div className="row">
            <label className="btn">
              {busy ? 'Uploading…' : 'Upload images / videos'}
              <input type="file" multiple accept="image/*,video/*" hidden disabled={busy} onChange={(e) => upload(e.target.files)} />
            </label>
            <button className="btn danger" onClick={async () => { await api.clearNormal(scene.id); onSceneChanged() }}>Clear normal data</button>
          </div>
        </div>

        <div className="card stack">
          <h3>2 · Train detectors</h3>
          <div className="row" style={{ gap: 16 }}>
            {MODEL_NAMES.map((m) => (
              <label key={m} className="row small" style={{ gap: 6 }}>
                <input type="checkbox" checked={trainModels.includes(m)}
                  onChange={(e) => setTrainModels((s) => (e.target.checked ? [...s, m] : s.filter((x) => x !== m)))} />
                <span className="model-dot" style={{ background: modelColor(m) }} />{MODEL_LABEL[m]}
              </label>
            ))}
          </div>
          <div className="row">
            <label className="field">
              Epochs (autoencoder / temporal)
              <input type="number" min={1} max={500} placeholder="default" value={epochs} onChange={(e) => setEpochs(e.target.value)} style={{ width: 120 }} />
            </label>
            <button className="btn primary" disabled={!trainModels.length || tj?.status === 'running'} onClick={train}>Train</button>
          </div>
          <JobProgress job={tj} />
          {tj?.status === 'done' && Object.entries(tj.result).filter(([, v]) => v?.error).map(([k, v]) => (
            <div key={k} className="small" style={{ color: 'var(--critical)' }}>! {MODEL_LABEL[k]}: {v.error}</div>
          ))}
        </div>
      </div>

      <div className="grid cols-3">
        {MODEL_NAMES.map((m) => {
          const info = scene.models[m]
          return (
            <div key={m} className="card stack" style={{ borderTop: `3px solid ${modelColor(m)}` }}>
              <div className="between">
                <h3>{MODEL_LABEL[m]}</h3>
                <span className="badge">
                  <span className="dot" style={{ background: info.trained ? 'var(--good)' : 'var(--text-muted)' }} />
                  {info.trained ? '✓ trained' : '– untrained'}
                </span>
              </div>
              <div className="small secondary">{info.title}</div>
              <div className="small secondary">{MODEL_BLURB[m]}</div>
              {info.trained && (
                <table className="data small">
                  <tbody>
                    {Object.entries(info.meta).map(([k, v]) => (
                      <tr key={k}><td className="muted">{k}</td><td className="num mono">{typeof v === 'number' ? +v.toFixed(4) : String(v)}</td></tr>
                    ))}
                    {info.calibration && (
                      <tr><td className="muted">threshold (raw)</td><td className="num mono">{info.calibration.img_hi.toPrecision(4)}</td></tr>
                    )}
                  </tbody>
                </table>
              )}
              {m === 'embedding' && info.trained && info.meta.pretrained === false && (
                <div className="small" style={{ color: 'var(--critical)' }}>! Backbone ran without pretrained weights (offline?) — accuracy will be poor.</div>
              )}
            </div>
          )
        })}
      </div>

      <div className="card stack">
        <h3>Analyse a video file</h3>
        <p className="small secondary" style={{ margin: 0 }}>Runs every trained detector over a recorded video, tracks anomalies over time and produces a session report.</p>
        <div className="row">
          <label className="field">
            Sample rate
            <select value={vfps} onChange={(e) => setVfps(Number(e.target.value))}>
              {[2, 5, 10, 15, 25].map((f) => <option key={f} value={f}>{f} fps</option>)}
            </select>
          </label>
          <label className="btn">
            Choose video…
            <input type="file" accept="video/*" hidden onChange={(e) => analyzeVideo(e.target.files?.[0])} />
          </label>
        </div>
        <JobProgress job={vj} />
        {vj?.status === 'done' && vj.result.session_id && (
          <button className="btn" onClick={() => onOpenSession(vj.result.session_id)}>Open report for session #{vj.result.session_id} →</button>
        )}
      </div>

      <div className="card">
        <h3>Recent jobs</h3>
        {jobs.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>#</th><th>Kind</th><th>Status</th><th>Progress</th><th>Message</th><th>Created</th></tr></thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.id}>
                    <td>{j.id}</td><td>{j.kind}</td><td>{j.status}</td><td className="num">{Math.round(j.progress * 100)}%</td>
                    <td className="small secondary" style={{ maxWidth: 420 }}>{j.message}</td><td className="small">{fmtTime(j.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty">No jobs yet.</div>}
      </div>
    </div>
  )
}

const MODEL_BLURB: Record<string, string> = {
  embedding: 'Stores pretrained-CNN features of normal patches; flags patches far from every stored one. Strong on subtle, static, local defects. Fast to fit (no training).',
  autoencoder: 'Learns to compress and rebuild normal images; flags what it cannot rebuild. Trains from scratch; tends to blur fine texture.',
  temporal: 'Predicts the next frame from the last few; flags unexpected motion and sudden appearances. Blind to anomalies that sit perfectly still.',
}
