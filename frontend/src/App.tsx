import { useCallback, useEffect, useState } from 'react'
import { ErrorLine } from './components/common'
import { api, type Scene } from './lib/api'
import Events from './pages/Events'
import Inspect from './pages/Inspect'
import Live from './pages/Live'
import Reports from './pages/Reports'
import Research from './pages/Research'
import Setup from './pages/Setup'

const TABS = ['Live', 'Inspect', 'Setup', 'Events', 'Reports', 'Research'] as const
type Tab = (typeof TABS)[number]

function store(key: string, value?: string): string | null {
  try {
    if (value !== undefined) localStorage.setItem(key, value)
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (TABS.includes(store('chimera.tab') as Tab) ? (store('chimera.tab') as Tab) : 'Live'))
  const [scenes, setScenes] = useState<Scene[]>([])
  const [sceneId, setSceneId] = useState<string | null>(() => store('chimera.scene'))
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [health, setHealth] = useState<{ device: string; version: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [newName, setNewName] = useState('')

  const refresh = useCallback(async () => {
    try {
      const s = await api.scenes()
      setScenes(s)
      setError(null)
      setSceneId((cur) => (cur && s.some((x) => x.id === cur) ? cur : s[0]?.id ?? null))
    } catch (e) {
      setError(`Backend unreachable: ${(e as Error).message}`)
    }
  }, [])

  useEffect(() => {
    refresh()
    api.health().then(setHealth).catch(() => {})
  }, [refresh])
  useEffect(() => void store('chimera.tab', tab), [tab])
  useEffect(() => void (sceneId && store('chimera.scene', sceneId)), [sceneId])

  const scene = scenes.find((s) => s.id === sceneId) ?? null
  const openSession = (id: number) => {
    setSessionId(id)
    setTab('Reports')
  }

  async function create() {
    if (!newName.trim()) return
    try {
      const s = await api.createScene(newName.trim())
      setNewName('')
      await refresh()
      setSceneId(s.id)
      setTab('Setup')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <>
      <header className="topbar">
        <div className="brand">
          CHIMERA <small>visual anomaly detection</small>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t} className={`tab ${t === tab ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
        <div className="spacer" />
        <div className="row">
          <select value={sceneId ?? ''} onChange={(e) => setSceneId(e.target.value || null)} aria-label="Scene">
            {!scenes.length && <option value="">No scenes</option>}
            {scenes.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
          <input placeholder="New scene name" value={newName} onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && create()} style={{ width: 150 }} />
          <button className="btn" onClick={create} disabled={!newName.trim()}>Add</button>
          {scene && (
            <button className="btn danger" title="Delete scene, its models and events" onClick={async () => {
              if (!confirm(`Delete scene "${scene.name}" with its models, sessions and events?`)) return
              await api.deleteScene(scene.id)
              setSceneId(null)
              refresh()
            }}>Delete</button>
          )}
          {health && <span className="small muted">{health.device}</span>}
        </div>
      </header>
      <main>
        <ErrorLine error={error} />
        {tab === 'Live' && <Live scene={scene} onSceneChanged={refresh} />}
        {tab === 'Inspect' && <Inspect scene={scene} />}
        {tab === 'Setup' && <Setup scene={scene} onSceneChanged={refresh} onOpenSession={openSession} />}
        {tab === 'Events' && <Events scene={scene} onOpenSession={openSession} />}
        {tab === 'Reports' && <Reports scene={scene} sessionId={sessionId} setSessionId={setSessionId} />}
        {tab === 'Research' && <Research />}
      </main>
    </>
  )
}
