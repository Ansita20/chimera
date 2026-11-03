import { useEffect, useState } from 'react'
import { ErrorLine, ExplanationCard, fmtTime, modelColor } from '../components/common'
import { api, MODEL_LABEL, url, type EventT, type Scene } from '../lib/api'

export default function Events({ scene, onOpenSession }: { scene: Scene | null; onOpenSession: (id: number) => void }) {
  const [events, setEvents] = useState<EventT[]>([])
  const [minScore, setMinScore] = useState(1)
  const [allScenes, setAllScenes] = useState(false)
  const [open, setOpen] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = () =>
    api
      .events({ scene_id: allScenes ? undefined : scene?.id, min_score: minScore, limit: 200 })
      .then(setEvents)
      .catch((e) => setError(e.message))
  useEffect(() => {
    load()
  }, [scene?.id, minScore, allScenes]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="stack">
      <div className="card row" style={{ gap: 16 }}>
        <label className="field">
          Minimum peak score · {minScore.toFixed(1)}×
          <input type="range" min={1} max={5} step={0.1} value={minScore} onChange={(e) => setMinScore(Number(e.target.value))} />
        </label>
        <label className="row small" style={{ gap: 6 }}>
          <input type="checkbox" checked={allScenes} onChange={(e) => setAllScenes(e.target.checked)} /> All scenes
        </label>
        <button className="btn" onClick={load}>Refresh</button>
        <span className="muted small">{events.length} events</span>
      </div>
      <ErrorLine error={error} />
      {events.length ? (
        <div className="grid cols-3">
          {events.map((e) => (
            <div key={e.id} className="card event-card">
              {e.snapshot_url && <img src={url(e.snapshot_url)} alt={`event ${e.id}`} loading="lazy" />}
              <div className="between">
                <b>{e.summary || 'Deviation'}</b>
                <span className="mono small">{e.peak_score.toFixed(1)}×</span>
              </div>
              <div className="small muted">
                {fmtTime(e.started_at)} · {e.duration_s.toFixed(1)}s · ±{Math.round(e.uncertainty * 100)}% · {e.scene_id}
              </div>
              <div className="row small" style={{ gap: 10 }}>
                {e.models.map((m) => (
                  <span key={m} className="row" style={{ gap: 4 }}>
                    <span className="model-dot" style={{ background: modelColor(m) }} /> {MODEL_LABEL[m] ?? m}
                  </span>
                ))}
              </div>
              <div className="row">
                <button className="btn" onClick={() => setOpen(open === e.id ? null : e.id)}>{open === e.id ? 'Hide' : 'Why?'}</button>
                <button className="btn" onClick={() => onOpenSession(e.session_id)}>Session #{e.session_id}</button>
              </div>
              {open === e.id && e.explanation?.reasons && <ExplanationCard e={e.explanation} />}
            </div>
          ))}
        </div>
      ) : (
        <div className="card empty">No events yet. Events are created when an anomalous region persists across frames (live or video) or is found in an uploaded image.</div>
      )}
    </div>
  )
}
