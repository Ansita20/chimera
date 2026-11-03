import { useCallback, useRef, useState, type ReactNode } from 'react'

export interface Series {
  key: string
  label: string
  color: string
  values: (number | null)[]
  dash?: string
}

const PAD = { l: 36, r: 12, t: 10, b: 24 }

function useWidth(): [(el: HTMLDivElement | null) => void, number] {
  const [w, setW] = useState(600)
  const ro = useRef<ResizeObserver | null>(null)
  const ref = useCallback((el: HTMLDivElement | null) => {
    ro.current?.disconnect()
    if (!el) return
    setW(el.clientWidth)
    if (typeof ResizeObserver === 'undefined') return
    ro.current = new ResizeObserver((e) => setW(e[0].contentRect.width))
    ro.current.observe(el)
  }, [])
  return [ref, w]
}

function spreadLabels<T extends { y: number }>(items: T[], gap = 13): T[] {
  const out = [...items].sort((a, b) => a.y - b.y).map((i) => ({ ...i }))
  for (let i = 1; i < out.length; i++) if (out[i].y - out[i - 1].y < gap) out[i].y = out[i - 1].y + gap
  return out
}

function niceMax(v: number): number {
  if (v <= 0) return 1
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p
  return 10 * p
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean; dash?: string; line?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}>
          {i.dashed || i.line ? (
            <svg width="22" height="4"><line x1="0" y1="2" x2="22" y2="2" stroke={i.color} strokeWidth="2" strokeDasharray={i.dash ?? (i.dashed ? '4 3' : undefined)} /></svg>
          ) : (
            <span className="model-dot" style={{ background: i.color }} />
          )}
          {i.label}
        </span>
      ))}
    </div>
  )
}

/** Multi-series line chart on ONE y-axis, with crosshair + tooltip and optional reference line. */
export function LineChart({
  x,
  series,
  height = 180,
  yMax,
  refLine,
  xFormat = (v) => String(v),
  yFormat = (v) => v.toFixed(2),
  directLabels = true,
}: {
  x: number[]
  series: Series[]
  height?: number
  yMax?: number
  refLine?: { y: number; label: string }
  xFormat?: (v: number) => string
  yFormat?: (v: number) => string
  directLabels?: boolean
}) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState<number | null>(null)
  const w = Math.max(200, width)
  const iw = w - PAD.l - PAD.r - (directLabels && series.length > 1 ? 118 : 0)
  const ih = height - PAD.t - PAD.b
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null))
  const top = yMax ?? niceMax(Math.max(refLine ? refLine.y * 1.2 : 0, ...all, 0.01))
  const x0 = x[0] ?? 0
  const x1 = x.length > 1 ? x[x.length - 1] : x0 + 1
  const sx = (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * iw
  const sy = (v: number) => PAD.t + ih - (Math.min(v, top) / top) * ih
  const ticks = [0, top / 2, top]

  const path = (vals: (number | null)[]) => {
    let d = ''
    let pen = false
    vals.forEach((v, i) => {
      if (v === null || x[i] === undefined) {
        pen = false
        return
      }
      d += `${pen ? 'L' : 'M'}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`
      pen = true
    })
    return d
  }

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    if (!x.length) return
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect()
    const px = e.clientX - r.left
    const target = x0 + (px / iw) * (x1 - x0)
    let best = 0
    for (let i = 1; i < x.length; i++) if (Math.abs(x[i] - target) < Math.abs(x[best] - target)) best = i
    setHover(best)
  }

  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={height} role="img" aria-label={series.map((s) => s.label).join(', ')}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={PAD.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={PAD.l - 6} y={sy(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {x.length > 1 &&
          [x0, (x0 + x1) / 2, x1].map((t, i) => (
            <text key={i} x={sx(t)} y={height - 6} textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}>
              {xFormat(t)}
            </text>
          ))}
        {refLine && (
          <g>
            <line x1={PAD.l} x2={PAD.l + iw} y1={sy(refLine.y)} y2={sy(refLine.y)} stroke="var(--text-secondary)" strokeDasharray="4 3" />
            <text x={PAD.l + iw} y={sy(refLine.y) - 4} textAnchor="end">{refLine.label}</text>
          </g>
        )}
        {series.map((s) => (
          <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeDasharray={s.dash} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {directLabels && series.length > 1 &&
          spreadLabels(
            series.flatMap((s) => {
              const i = s.values.length - 1 - [...s.values].reverse().findIndex((v) => v !== null)
              return i < s.values.length && s.values[i] !== null ? [{ key: s.key, label: s.label, y: sy(s.values[i] as number) }] : []
            }),
          ).map((l) => (
            <text key={l.key} x={PAD.l + iw + 6} y={l.y + 4} style={{ fill: 'var(--text-secondary)' }}>
              {l.label}
            </text>
          ))}
        {hover !== null && x[hover] !== undefined && (
          <g>
            <line x1={sx(x[hover])} x2={sx(x[hover])} y1={PAD.t} y2={PAD.t + ih} stroke="var(--text-muted)" />
            {series.map((s) =>
              s.values[hover] !== null && s.values[hover] !== undefined ? (
                <circle key={s.key} cx={sx(x[hover])} cy={sy(s.values[hover] as number)} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />
              ) : null,
            )}
          </g>
        )}
        <rect x={PAD.l} y={PAD.t} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </svg>
      {hover !== null && x[hover] !== undefined && (
        <div className="tip" style={{ left: Math.min(sx(x[hover]) + 12, w - 160), top: PAD.t }}>
          <div className="muted">{xFormat(x[hover])}</div>
          {series.map((s) => (
            <div key={s.key} className="row" style={{ gap: 6 }}>
              <span className="model-dot" style={{ background: s.color }} />
              {s.label}: <b>{s.values[hover] === null ? '–' : yFormat(s.values[hover] as number)}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/** Single-series vertical bars with per-bar hover tooltip. */
export function BarChart({
  data,
  color = '#555',
  height = 160,
  format = (v: number) => String(v),
  labelEvery = 1,
}: {
  data: { label: string; value: number }[]
  color?: string
  height?: number
  format?: (v: number) => string
  labelEvery?: number
}) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState<number | null>(null)
  const w = Math.max(200, width)
  const iw = w - PAD.l - PAD.r
  const ih = height - PAD.t - PAD.b
  const top = niceMax(Math.max(...data.map((d) => d.value), 1))
  const bw = iw / Math.max(1, data.length)
  const gap = Math.min(2 + bw * 0.25, 12)
  const sy = (v: number) => PAD.t + ih - (v / top) * ih
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={height} role="img">
        {[0, top / 2, top].map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={PAD.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={PAD.l - 6} y={sy(t) + 4} textAnchor="end">{format(t)}</text>
          </g>
        ))}
        {data.map((d, i) => {
          const x = PAD.l + i * bw + gap / 2
          const bh = Math.max(0, (d.value / top) * ih)
          const r = Math.min(4, bh / 2, (bw - gap) / 2)
          const y = PAD.t + ih - bh
          const bwi = Math.max(1, bw - gap)
          return (
            <g key={d.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={PAD.l + i * bw} y={PAD.t} width={bw} height={ih} fill="transparent" />
              {bh > 0 && (
                <path
                  d={`M${x},${y + bh}V${y + r}Q${x},${y} ${x + r},${y}H${x + bwi - r}Q${x + bwi},${y} ${x + bwi},${y + r}V${y + bh}Z`}
                  fill={color}
                  opacity={hover === null || hover === i ? 1 : 0.55}
                />
              )}
              {i % labelEvery === 0 && (
                <text x={x + bwi / 2} y={height - 6} textAnchor="middle">{d.label}</text>
              )}
            </g>
          )
        })}
      </svg>
      {hover !== null && (
        <div className="tip" style={{ left: Math.min(PAD.l + hover * bw + bw, w - 120), top: 0 }}>
          {data[hover].label}: <b>{format(data[hover].value)}</b>
        </div>
      )}
    </div>
  )
}

/** Grouped bars: categories on x, one bar per series, one y-axis. */
export function GroupedBars({
  categories,
  series,
  height = 220,
  yMax = 1,
  format = (v: number) => v.toFixed(3),
}: {
  categories: string[]
  series: Series[]
  height?: number
  yMax?: number
  format?: (v: number) => string
}) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState<[number, number] | null>(null)
  const w = Math.max(240, width)
  const iw = w - PAD.l - PAD.r
  const ih = height - PAD.t - PAD.b
  const gw = iw / Math.max(1, categories.length)
  const inner = gw * 0.8
  const bw = Math.max(2, inner / Math.max(1, series.length) - 2)
  const sy = (v: number) => PAD.t + ih - (Math.min(v, yMax) / yMax) * ih
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={height} role="img">
        {[0, yMax / 2, yMax].map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={PAD.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={PAD.l - 6} y={sy(t) + 4} textAnchor="end">{t.toFixed(1)}</text>
          </g>
        ))}
        {categories.map((c, ci) => (
          <g key={c}>
            {series.map((s, si) => {
              const v = s.values[ci]
              if (v === null || v === undefined) return null
              const x = PAD.l + ci * gw + (gw - inner) / 2 + si * (bw + 2)
              const y = sy(v)
              const bh = PAD.t + ih - y
              const r = Math.min(4, bw / 2, bh / 2)
              return (
                <path
                  key={s.key}
                  d={`M${x},${y + bh}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${y + bh}Z`}
                  fill={s.color}
                  opacity={hover === null || (hover[0] === ci && hover[1] === si) ? 1 : 0.5}
                  onMouseEnter={() => setHover([ci, si])}
                  onMouseLeave={() => setHover(null)}
                />
              )
            })}
            <text x={PAD.l + ci * gw + gw / 2} y={height - 6} textAnchor="middle">{c}</text>
          </g>
        ))}
      </svg>
      {hover && (
        <div className="tip" style={{ left: Math.min(PAD.l + hover[0] * gw + gw / 2, w - 180), top: 0 }}>
          {categories[hover[0]]} · {series[hover[1]].label}: <b>{format(series[hover[1]].values[hover[0]] as number)}</b>
        </div>
      )}
    </div>
  )
}

export function HBars({ data, color = '#555', format = (v: number) => String(v) }: {
  data: { label: ReactNode; value: number; key: string }[]
  color?: string
  format?: (v: number) => string
}) {
  const max = Math.max(1, ...data.map((d) => d.value))
  if (!data.length) return <div className="empty">No data yet</div>
  return (
    <div className="stack" style={{ gap: 6 }}>
      {data.map((d) => (
        <div className="hbar" key={d.key} title={`${d.value}`}>
          <span className="secondary">{d.label}</span>
          <div className="bar-track" style={{ height: 10 }}>
            <div className="bar-fill" style={{ width: `${(d.value / max) * 100}%`, background: color }} />
          </div>
          <span className="num">{format(d.value)}</span>
        </div>
      ))}
    </div>
  )
}
