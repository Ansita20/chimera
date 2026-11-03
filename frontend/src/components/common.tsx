import { useEffect, useState, type ReactNode } from 'react'
import { api, MODEL_LABEL, type Explanation, type Job, type ModelScore, type Region, type Verdict } from '../lib/api'

export const modelColor = (m: string) => `var(--m-${m}, var(--text-muted))`
const DASH: Record<string, string> = { embedding: '7 3', autoencoder: '2 3', temporal: '10 3 2 3' }
export const modelDash = (m: string) => DASH[m]

const VERDICT: Record<Verdict, { label: string; icon: string; color: string }> = {
  normal: { label: 'Normal', icon: '✓', color: 'var(--good)' },
  anomalous: { label: 'Anomaly', icon: '!', color: 'var(--critical)' },
  uncertain: { label: 'Uncertain', icon: '?', color: 'var(--warning)' },
  warming_up: { label: 'Warming up', icon: '…', color: 'var(--text-muted)' },
  idle: { label: 'Idle', icon: '–', color: 'var(--text-muted)' },
}

export function VerdictBadge({ verdict, big }: { verdict: Verdict; big?: boolean }) {
  const v = VERDICT[verdict]
  return (
    <span className={`badge ${big ? 'big' : ''}`} role="status">
      <span className="dot" style={{ background: v.color }} />
      <span aria-hidden>{v.icon}</span> {v.label}
    </span>
  )
}

export function Tile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub !== undefined && <div className="sub">{sub}</div>}
    </div>
  )
}

/** Score bar on a 0–3× scale with a tick at the threshold. */
export function ScoreBar({ score, threshold = 1, color }: { score: number; threshold?: number; color: string }) {
  const max = Math.max(3, threshold * 1.5)
  return (
    <div className="bar-track" aria-hidden>
      <div className="bar-fill" style={{ width: `${Math.min(100, (score / max) * 100)}%`, background: color }} />
      <div className="bar-thr" style={{ left: `${(threshold / max) * 100}%` }} />
    </div>
  )
}

export function ModelScores({ models, threshold = 1 }: { models: Record<string, ModelScore>; threshold?: number }) {
  const entries = Object.entries(models)
  if (!entries.length) return <div className="muted small">No model output yet</div>
  return (
    <div className="stack" style={{ gap: 8 }}>
      {entries.map(([m, s]) => (
        <div key={m} className="stack" style={{ gap: 4 }}>
          <div className="between small">
            <span className="row" style={{ gap: 6 }}>
              <span className="model-dot" style={{ background: modelColor(m) }} />
              {MODEL_LABEL[m] ?? m}
            </span>
            <span className="mono">
              {(s.score / threshold).toFixed(2)}× · ±{Math.round(s.uncertainty * 100)}%
            </span>
          </div>
          <ScoreBar score={s.score} threshold={threshold} color={modelColor(m)} />
        </div>
      ))}
    </div>
  )
}

export function ExplanationCard({ e, title }: { e: Explanation; title?: ReactNode }) {
  return (
    <div className="expl stack" style={{ gap: 8 }}>
      <div className="between">
        <b>{title ?? e.summary}</b>
        <span className="small muted">confidence {Math.round(e.confidence * 100)}%</span>
      </div>
      {title && <div className="small secondary">{e.summary}</div>}
      {e.reasons.map((r, i) => (
        <div key={i} className="reason" style={{ borderLeftColor: r.factor in FACTOR_MODEL ? modelColor(FACTOR_MODEL[r.factor]) : 'var(--border)' }}>
          <div className="tag">{r.tag}</div>
          <p>{r.text}</p>
        </div>
      ))}
      <div className="small muted">
        Describes how this region deviates from learned normality — not what the object is.
      </div>
    </div>
  )
}

const FACTOR_MODEL: Record<string, string> = { novel_appearance: 'embedding', reconstruction: 'autoencoder', motion: 'temporal' }

export function RegionBoxes({ regions }: { regions: Region[] }) {
  return (
    <svg className="boxes" viewBox="0 0 100 100" preserveAspectRatio="none">
      {regions.map((r, i) => {
        const [x, y, w, h] = r.bbox
        return (
          <g key={r.track_id ?? i}>
            <rect
              x={x * 100}
              y={y * 100}
              width={w * 100}
              height={h * 100}
              fill="none"
              stroke="#ffffff"
              strokeWidth={r.confirmed ? 2 : 1.5}
              strokeDasharray={r.confirmed ? undefined : '3 2'}
              vectorEffect="non-scaling-stroke"
            />
          </g>
        )
      })}
    </svg>
  )
}

export function RegionLabels({ regions }: { regions: Region[] }) {
  return (
    <>
      {regions.map((r, i) => (
        <div
          key={r.track_id ?? i}
          style={{
            position: 'absolute',
            left: `${r.bbox[0] * 100}%`,
            top: `calc(${r.bbox[1] * 100}% - 20px)`,
            background: 'rgba(0,0,0,.7)',
            color: '#fff',
            fontSize: 11,
            lineHeight: '16px',
            padding: '1px 6px',
            borderRadius: 4,
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
          }}
        >
          {r.track_id ? `#${r.track_id} ` : ''}
          {r.peak.toFixed(1)}× · {r.explanation.summary}
        </div>
      ))}
    </>
  )
}

/** Poll a background job until it finishes. */
export function useJob(jobId: number | null, onDone?: (j: Job) => void) {
  const [job, setJob] = useState<Job | null>(null)
  useEffect(() => {
    if (jobId === null) return
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      try {
        const j = await api.job(jobId)
        if (!alive) return
        setJob(j)
        if (j.status === 'done' || j.status === 'error') {
          onDone?.(j)
          return
        }
      } catch {
        /* transient */
      }
      timer = setTimeout(tick, 1000)
    }
    tick()
    return () => {
      alive = false
      clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])
  return job
}

export function JobProgress({ job }: { job: Job | null }) {
  if (!job) return null
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="between small">
        <span>
          {job.kind} #{job.id} — <b>{job.status}</b>
        </span>
        <span className="mono">{Math.round(job.progress * 100)}%</span>
      </div>
      <div className="progress">
        <div style={{ width: `${job.progress * 100}%`, background: job.status === 'error' ? 'var(--critical)' : undefined }} />
      </div>
      <div className="small muted" style={{ wordBreak: 'break-word' }}>{job.message}</div>
    </div>
  )
}

export function ErrorLine({ error }: { error: string | null }) {
  if (!error) return null
  return (
    <div className="alert">
      <span className="icon" style={{ color: 'var(--critical)' }}>!</span>
      <span>{error}</span>
    </div>
  )
}

export const fmtTime = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '–')
export const pct = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`)
