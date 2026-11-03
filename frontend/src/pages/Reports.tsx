import { useEffect, useState } from 'react'
import { BarChart, HBars, Legend, LineChart } from '../components/Charts'
import { ErrorLine, Tile, fmtTime, modelColor, modelDash, pct } from '../components/common'
import { api, MODEL_LABEL, MODEL_NAMES, url, type Overview, type Scene, type SessionReport, type SessionT } from '../lib/api'

const FACTOR_LABEL: Record<string, string> = {
  novel_appearance: 'Unfamiliar appearance',
  reconstruction: 'Unreconstructable',
  motion: 'Unexpected change',
  brightness: 'Brightness',
  color: 'Colour shift',
  texture: 'Texture',
  global: 'Scene-wide change',
  persistence: 'Persistence',
  movement: 'Moving',
  uncertainty: 'Low confidence',
  disagreement: 'Partial agreement',
}

export default function Reports({ scene, sessionId, setSessionId }: {
  scene: Scene | null
  sessionId: number | null
  setSessionId: (id: number | null) => void
}) {
  const [days, setDays] = useState(14)
  const [ov, setOv] = useState<Overview | null>(null)
  const [sessions, setSessions] = useState<SessionT[]>([])
  const [rep, setRep] = useState<SessionReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.overview(scene?.id, days).then(setOv).catch((e) => setError(e.message))
    api.sessions(scene?.id).then(setSessions).catch(() => {})
  }, [scene?.id, days])
  useEffect(() => {
    if (sessionId === null) return setRep(null)
    api.report(sessionId).then(setRep).catch((e) => setError(e.message))
  }, [sessionId])

  return (
    <div className="stack">
      <ErrorLine error={error} />
      <div className="between">
        <h2>{scene ? `Overview · ${scene.name}` : 'Overview'}</h2>
        <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
          {[1, 7, 14, 30, 90].map((d) => <option key={d} value={d}>Last {d} day{d > 1 ? 's' : ''}</option>)}
        </select>
      </div>
      {ov && (
        <>
          <div className="tiles">
            <Tile label="Events" value={ov.totals.events} />
            <Tile label="Sessions" value={ov.totals.sessions} />
            <Tile label="Mean uncertainty" value={pct(ov.totals.mean_uncertainty)} />
            {MODEL_NAMES.map((m) => (
              <Tile key={m} label={`Flagged by ${MODEL_LABEL[m]}`} value={ov.by_model[m] ?? 0} />
            ))}
          </div>
          <div className="grid cols-2">
            <div className="card">
              <h3>Events per day</h3>
              <BarChart data={ov.events_per_day.map((d) => ({ label: d.date.slice(5), value: d.events }))}
                labelEvery={Math.ceil(ov.events_per_day.length / 8)} />
            </div>
            <div className="card">
              <h3>Peak score distribution</h3>
              <BarChart data={ov.score_hist.map((b) => ({ label: b.bin, value: b.count }))} />
            </div>
            <div className="card">
              <h3>Why regions were flagged</h3>
              <HBars data={Object.entries(ov.reasons).map(([k, v]) => ({ key: k, label: FACTOR_LABEL[k] ?? k, value: v }))} />
            </div>
            <div className="card">
              <h3>Recent events</h3>
              {ov.recent.length ? (
                <div className="thumbs">
                  {ov.recent.slice(0, 6).map((e) => (
                    <div key={e.id} className="stack" style={{ gap: 4, cursor: 'pointer' }} onClick={() => setSessionId(e.session_id)}>
                      {e.snapshot_url && <img src={url(e.snapshot_url)} alt="" style={{ width: '100%', borderRadius: 6 }} />}
                      <span className="small"><b>{e.peak_score.toFixed(1)}×</b> {e.summary}</span>
                    </div>
                  ))}
                </div>
              ) : <div className="empty">No events in this period.</div>}
            </div>
          </div>
        </>
      )}

      <div className="card">
        <h3>Sessions</h3>
        {sessions.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>#</th><th>Source</th><th>Started</th><th className="num">Frames</th><th className="num">Anomalous</th><th className="num">Max score</th></tr></thead>
              <tbody>
                {sessions.map((s) => (
                  <tr key={s.id} className={`clickable ${s.id === sessionId ? 'sel' : ''}`} onClick={() => setSessionId(s.id)}>
                    <td>{s.id}</td><td>{s.source}</td><td className="small">{fmtTime(s.started_at)}</td>
                    <td className="num">{s.frames}</td><td className="num">{s.frames ? pct(s.anomalous_frames / s.frames) : '–'}</td>
                    <td className="num">{s.max_score.toFixed(2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="empty">No sessions yet.</div>}
      </div>

      {rep && <SessionView rep={rep} />}
    </div>
  )
}

function SessionView({ rep }: { rep: SessionReport }) {
  const tl = rep.timeline
  const x = tl.map((p) => p.t)
  const models = MODEL_NAMES.filter((m) => tl.some((p) => p[`m_${m}`] !== undefined))
  const fmtT = (v: number) => `${v.toFixed(1)}s`
  return (
    <div className="card stack">
      <div className="between">
        <h3>Session #{rep.session.id} · {rep.session.source}</h3>
        <span className="small muted">{fmtTime(rep.session.started_at)}</span>
      </div>
      <div className="tiles">
        <Tile label="Frames" value={rep.stats.frames} />
        <Tile label="Anomalous frames" value={pct(rep.stats.anomalous_ratio)} />
        <Tile label="Max score" value={`${rep.stats.max_score.toFixed(2)}×`} />
        <Tile label="Mean uncertainty" value={pct(rep.stats.mean_uncertainty)} />
        <Tile label="Events" value={rep.stats.events} />
      </div>
      {tl.length > 1 ? (
        <>
          <div className="between">
            <h3>Score timeline</h3>
            <Legend items={[{ label: 'Fused', color: 'var(--text-primary)', line: true }, ...models.map((m) => ({ label: MODEL_LABEL[m], color: modelColor(m), dash: modelDash(m), line: true }))]} />
          </div>
          <LineChart x={x} xFormat={fmtT} yFormat={(v) => `${v.toFixed(1)}×`} refLine={{ y: 1, label: 'threshold' }} height={220}
            series={[
              { key: 'f', label: 'Fused', color: 'var(--text-primary)', values: tl.map((p) => p.score) },
              ...models.map((m) => ({ key: m, label: MODEL_LABEL[m], color: modelColor(m), dash: modelDash(m), values: tl.map((p) => p[`m_${m}`] ?? null) })),
            ]} />
          <h3>Uncertainty timeline</h3>
          <LineChart x={x} xFormat={fmtT} yMax={1} yFormat={(v) => `${Math.round(v * 100)}%`} height={110}
            series={[{ key: 'u', label: 'Uncertainty', color: 'var(--text-secondary)', values: tl.map((p) => p.uncertainty) }]} />
        </>
      ) : <div className="muted small">Single-frame session — no timeline.</div>}
      <h3>Events</h3>
      {rep.events.length ? (
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th></th><th>Track</th><th>Summary</th><th className="num">Start</th><th className="num">Duration</th><th className="num">Peak</th><th className="num">Uncertainty</th><th>Detectors</th></tr></thead>
            <tbody>
              {rep.events.map((e) => (
                <tr key={e.id}>
                  <td style={{ width: 96 }}>{e.snapshot_url && <img src={url(e.snapshot_url)} alt="" style={{ width: 88, borderRadius: 4, display: 'block' }} />}</td>
                  <td>#{e.track_id}</td><td>{e.summary}</td><td className="num">f{e.frame_start}</td>
                  <td className="num">{e.duration_s.toFixed(1)}s</td><td className="num">{e.peak_score.toFixed(2)}×</td>
                  <td className="num">{pct(e.uncertainty)}</td>
                  <td>{e.models.map((m) => <span key={m} className="model-dot" title={MODEL_LABEL[m]} style={{ background: modelColor(m), marginRight: 4 }} />)}
                    <span className="small muted">{e.models.map((m) => MODEL_LABEL[m] ?? m).join(', ')}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div className="muted small">No events in this session.</div>}
    </div>
  )
}
