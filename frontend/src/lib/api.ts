export const API_BASE: string = import.meta.env.VITE_API_URL ?? ''

export type ModelName = 'embedding' | 'autoencoder' | 'temporal'
export const MODEL_NAMES: ModelName[] = ['embedding', 'autoencoder', 'temporal']
export const MODEL_LABEL: Record<string, string> = {
  embedding: 'Feature embedding',
  autoencoder: 'Autoencoder',
  temporal: 'Temporal',
}

export interface ModelInfo {
  title: string
  trained: boolean
  meta: Record<string, number | string | boolean>
  calibration: Record<string, number> | null
}

export interface Scene {
  id: string
  name: string
  description: string
  created_at: string
  models: Record<ModelName, ModelInfo>
  normal: { images: number; sequences: number; sequence_frames: number }
  events: number
}

export interface Job {
  id: number
  kind: string
  scene_id: string | null
  status: 'queued' | 'running' | 'done' | 'error'
  progress: number
  message: string
  params: Record<string, unknown>
  result: Record<string, any>
  created_at: string
  finished_at: string | null
}

export interface Reason {
  factor: string
  tag: string
  text: string
  strength: number
}

export interface Explanation {
  summary: string
  reasons: Reason[]
  confidence: number
  peak: number
  models_flagged: string[]
  kind: string
}

export interface Region {
  track_id: number | null
  confirmed: boolean
  bbox: [number, number, number, number]
  area_frac: number
  peak: number
  model_peaks: Record<string, number>
  age_s: number
  hits: number
  explanation: Explanation
}

export interface ModelScore {
  score: number
  raw_score: number
  uncertainty: number
  epistemic: number
  ambiguity: number
  anomalous: boolean
}

export type Verdict = 'normal' | 'anomalous' | 'uncertain' | 'warming_up' | 'idle'

export interface FrameResult {
  frame_idx: number
  ts: number
  score: number
  uncertainty: number
  verdict: Verdict
  models: Record<string, ModelScore>
  regions: Region[]
  heatmap: string
  drift: { ema: number; alert: boolean }
  latency_ms: number
  model_heatmaps?: Record<string, string>
  reconstruction?: string
  events_new?: { id: number; track_id: number; summary: string; peak: number }[]
  session_id?: number | null
  width?: number
  height?: number
}

export interface EventT {
  id: number
  session_id: number
  scene_id: string
  track_id: number
  started_at: string
  duration_s: number
  frame_start: number
  frame_end: number
  peak_score: number
  uncertainty: number
  bbox: number[]
  models: string[]
  summary: string
  explanation: Explanation
  status: string
  snapshot_url: string | null
}

export interface SessionT {
  id: number
  scene_id: string
  source: string
  config: Record<string, unknown>
  started_at: string
  ended_at: string | null
  frames: number
  anomalous_frames: number
  max_score: number
}

export interface SessionReport {
  session: SessionT
  timeline: ({ t: number; frame: number; score: number; uncertainty: number; regions: number } & Record<string, number>)[]
  events: EventT[]
  stats: {
    frames: number
    anomalous_ratio: number
    max_score: number
    mean_uncertainty: number | null
    events: number
    per_model_mean_score: Record<string, number>
    reasons: Record<string, number>
  }
}

export interface Overview {
  days: number
  totals: { events: number; sessions: number; scenes: number; mean_uncertainty: number | null }
  events_per_day: { date: string; events: number }[]
  by_scene: Record<string, number>
  by_model: Record<string, number>
  reasons: Record<string, number>
  score_hist: { bin: string; count: number }[]
  recent: EventT[]
}

export interface BenchRun {
  id: number
  dataset: string
  category: string
  created_at: string
  results: Record<string, any>
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(API_BASE + path, init)
  if (!r.ok) {
    let msg = r.statusText
    try {
      const body = await r.json()
      msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* not json */
    }
    throw new Error(msg)
  }
  return (r.status === 204 ? undefined : await r.json()) as T
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

function form(fields: Record<string, string | Blob | undefined | null>, files?: [string, File][]) {
  const f = new FormData()
  for (const [k, v] of Object.entries(fields)) if (v !== undefined && v !== null) f.append(k, v)
  for (const [k, file] of files ?? []) f.append(k, file)
  return { method: 'POST', body: f }
}

export const api = {
  health: () => req<{ status: string; version: string; device: string; backbone: string }>('/api/health'),
  scenes: () => req<Scene[]>('/api/scenes'),
  scene: (id: string) => req<Scene>(`/api/scenes/${id}`),
  createScene: (name: string, description = '') => req<Scene>('/api/scenes', json('POST', { name, description })),
  deleteScene: (id: string) => req<void>(`/api/scenes/${id}`, { method: 'DELETE' }),
  uploadNormal: (id: string, files: File[]) =>
    req<{ added: Record<string, number>; errors: string[] }>(
      `/api/scenes/${id}/normal`,
      form({}, files.map((f) => ['files', f] as [string, File])),
    ),
  clearNormal: (id: string) => req(`/api/scenes/${id}/normal`, { method: 'DELETE' }),
  train: (id: string, models: string[], epochs?: number) =>
    req<{ job_id: number }>(`/api/scenes/${id}/train`, json('POST', { models, epochs: epochs || null })),
  detect: (id: string, file: File, o: { models?: string[]; threshold?: number; fusion?: string; save?: boolean }) =>
    req<FrameResult>(
      `/api/scenes/${id}/detect`,
      form(
        {
          models: o.models?.join(','),
          threshold: String(o.threshold ?? 1),
          fusion: o.fusion ?? 'max',
          save: String(o.save ?? true),
        },
        [['file', file]],
      ),
    ),
  analyzeVideo: (id: string, file: File, o: { models?: string[]; threshold?: number; fps?: number }) =>
    req<{ job_id: number }>(
      `/api/scenes/${id}/videos`,
      form({ models: o.models?.join(','), threshold: String(o.threshold ?? 1), fps: String(o.fps ?? 10) }, [['file', file]]),
    ),
  job: (id: number) => req<Job>(`/api/jobs/${id}`),
  jobs: (sceneId?: string) => req<Job[]>(`/api/jobs${sceneId ? `?scene_id=${sceneId}` : ''}`),
  sessions: (sceneId?: string) => req<SessionT[]>(`/api/sessions${sceneId ? `?scene_id=${sceneId}` : ''}`),
  report: (id: number) => req<SessionReport>(`/api/sessions/${id}/report`),
  events: (p: { scene_id?: string; session_id?: number; min_score?: number; limit?: number }) => {
    const q = new URLSearchParams()
    for (const [k, v] of Object.entries(p)) if (v !== undefined && v !== '') q.set(k, String(v))
    return req<EventT[]>(`/api/events?${q}`)
  },
  overview: (sceneId: string | undefined, days: number) =>
    req<Overview>(`/api/reports/overview?days=${days}${sceneId ? `&scene_id=${sceneId}` : ''}`),
  benchmarks: () => req<BenchRun[]>('/api/benchmarks'),
  runBenchmark: (body: Record<string, unknown>) => req<{ job_id: number }>('/api/benchmarks', json('POST', body)),
}

export const url = (p: string | null | undefined) => (p ? API_BASE + p : '')

export function wsUrl(path: string): string {
  const base = API_BASE || window.location.origin
  return base.replace(/^http/, 'ws') + path
}
