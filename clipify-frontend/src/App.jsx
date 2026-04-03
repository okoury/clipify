import { useState, useRef, useCallback, useEffect, useMemo, Component } from 'react'
import './styles.css'

// ══════════════════════════════════════════════════════════
// HELPERS & CONSTANTS
// ══════════════════════════════════════════════════════════

function chunkWords(words, size = 5) {
  const out = [], cur = []
  for (let i = 0; i < words.length; i++) {
    const gap = i > 0 ? words[i].start - words[i - 1].end : 0
    if (cur.length > 0 && (cur.length >= size || gap > 1.5)) { out.push([...cur]); cur.length = 0 }
    cur.push(words[i])
  }
  if (cur.length > 0) out.push([...cur])
  return out
}

const EMOJI_RULES = [
  [['crazy','insane','unbelievable','mindblowing','mind-blowing'],'🤯'],
  [['fire','hot','lit','flame'],                                  '🔥'],
  [['amazing','incredible','awesome','unreal'],                   '🔥'],
  [['love','loved','heart','adore'],                              '❤️'],
  [['laugh','laughing','funny','hilarious'],                      '😂'],
  [['money','cash','million','billion','rich','dollar'],          '💰'],
  [['win','won','winning','champion','victory'],                  '🏆'],
  [['secret','hidden','reveal','exposed','nobody knows'],         '🤫'],
  [['think','thinking','mind','brain','realize'],                 '🧠'],
  [['scared','fear','terrified','nightmare'],                     '😨'],
  [['angry','mad','furious','rage','hate'],                       '😤'],
  [['sad','cry','crying','tears','heartbreak'],                   '😢'],
  [['happy','joy','excited','celebrate','party'],                 '🎉'],
  [['look','see','watch','eyes','stare'],                         '👀'],
  [['truth','facts','real','honest','literally'],                 '💯'],
  [['dead','die','death','dying'],                                '💀'],
  [['strong','strength','power','muscle','grind'],                '💪'],
  [['run','running','sprint','race','fast'],                      '🏃'],
  [['eat','food','hungry','delicious','taste'],                   '😋'],
  [['music','song','beat','dance','vibe'],                        '🎵'],
  [['wow','omg','whoa','wait','seriously'],                       '😲'],
]

function pickEmoji(chunk) {
  const text = chunk.map(w => w.word.toLowerCase()).join(' ')
  for (const [kws, emoji] of EMOJI_RULES) {
    if (kws.some(k => text.includes(k))) return emoji
  }
  return null
}

// Censor map — stored as an opaque blob, not plaintext (mirrors the pipeline approach)
const CENSOR_MAP = JSON.parse(atob(
  'eyJmdWNrIjoiZioqayIsImZ1Y2tpbmciOiJmKipraW5nIiwiZnVja2VkIjoiZioqa2VkIiwiZnVja2VyIjoiZioqa2VyIiwiZnVja3MiOiJmKiprcyIsInNoaXQiOiJzaCp0Iiwic2hpdHMiOiJzaCp0cyIsInNoaXR0aW5nIjoic2gqdHRpbmciLCJzaGl0dHkiOiJzaCp0dHkiLCJidWxsc2hpdCI6ImJ1bGxzKip0IiwiYml0Y2giOiJiKipjaCIsImJpdGNoZXMiOiJiKipjaGVzIiwiYXNzIjoiYSoqIiwiYXNzZXMiOiJhKiplcyIsImFzc2hvbGUiOiJhKipob2xlIiwiYXNzaG9sZXMiOiJhKipob2xlcyIsImRpY2siOiJkKiprIiwiZGlja3MiOiJkKiprcyIsImNvY2siOiJjKiprIiwiY29ja3MiOiJjKiprcyIsImN1bnQiOiJjKip0IiwiY3VudHMiOiJjKip0cyIsInNleCI6InMqeCIsInNleHkiOiJzKnh5Iiwic2V4dWFsIjoicyp4dWFsIiwicGlzcyI6InAqc3MiLCJwaXNzZWQiOiJwKnNzZWQiLCJzbHV0Ijoic2wqdCIsInNsdXRzIjoic2wqdHMiLCJ3aG9yZSI6IndoKnJlIiwid2hvcmVzIjoid2gqcmVzIiwiYmFzdGFyZCI6ImIqc3RhcmQiLCJiYXN0YXJkcyI6ImIqc3RhcmRzIiwiZmFnIjoiZipnIiwiZmFnZ290IjoiZioqKip0IiwibmlnZ2VyIjoibioqKioqIiwibmlnZ2EiOiJuKioqYSIsImRhbW4iOiJkKm1uIiwiZGFtbmVkIjoiZCptbmVkIiwiaGVsbCI6ImgqbGwiLCJjcmFwIjoiY3IqcCIsInJldGFyZCI6InIqKipkIiwicmV0YXJkZWQiOiJyKioqZGVkIn0='
))

function censorWord(w) {
  const m = w.match(/^([^a-zA-Z]*)([a-zA-Z]+)([^a-zA-Z]*)$/)
  if (!m) return w
  const [, pre, core, post] = m
  const censored = CENSOR_MAP[core.toLowerCase()]
  return censored ? pre + censored + post : w
}
function censorText(t) { return t.replace(/\b\w+\b/g, w => CENSOR_MAP[w.toLowerCase()] ?? w) }

// Avatar helpers
const PALETTE = ['#6366f1','#8b5cf6','#ec4899','#f97316','#10b981','#3b82f6','#ef4444','#f59e0b']
const avatarColor = (n = '') => PALETTE[n.charCodeAt(0) % PALETTE.length] || PALETTE[0]
const initials    = (n = '') => n.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2) || '?'
const fmtDate     = (iso) => {
  if (!iso) return ''
  const d = new Date(iso.endsWith('Z') ? iso : iso + 'Z')
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

// API helper
const makeApi = (token) => async (path, opts = {}) => {
  const headers = { 'Content-Type': 'application/json', ...opts.headers }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(path, { ...opts, headers })
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `Error ${res.status}`) }
  return res.json()
}

// Status messages
const CLIP_STATUS = [
  'Analyzing your video…','Loading transcription model…','Transcribing audio…',
  'Scoring segments…','Finding highlight zones…','Extracting clips…','Almost there…',
]
const ANNOTATE_STATUS = [
  'Preparing video…','Loading transcription model…','Transcribing audio…','Generating captions…','Almost there…',
]

// URL routing
const ROUTE_MAP = { '/': 'home', '/about': 'about', '/history': 'history' }
const PATH_MAP  = { home: '/', about: '/about', history: '/history' }
const getViewFromPath = () => ROUTE_MAP[window.location.pathname] ?? '404'


// ══════════════════════════════════════════════════════════
// CAPTIONED VIDEO
// ══════════════════════════════════════════════════════════
// Caption font definitions (shared between FontPicker and CaptionedVideo)
const CAPTION_FONTS = [
  { id: 'impact',   label: 'Impact',   css: "'Impact','Haettenschweiler','Arial Narrow Bold',sans-serif" },
  { id: 'bold',     label: 'Bold',     css: "'Arial Black','Helvetica Neue',sans-serif" },
  { id: 'clean',    label: 'Clean',    css: "'Segoe UI','Helvetica Neue',Arial,sans-serif" },
  { id: 'serif',    label: 'Serif',    css: "Georgia,'Times New Roman',serif" },
  { id: 'mono',     label: 'Mono',     css: "'Courier New',Courier,monospace" },
  { id: 'rounded',  label: 'Rounded',  css: "'Trebuchet MS','Gill Sans',sans-serif" },
]

// Maps style IDs to CSS class names applied to active caption words
const HL_CLASS = {
  golden:   'hl-golden',
  neon:     'hl-neon',
  flamingo: 'hl-flamingo',
  electric: 'hl-electric',
  ember:    'hl-ember',
  violet:   'hl-violet',
  rainbow:  'hl-rainbow',
  subtitles:'hl-subtitles',
}

function CaptionedVideo({ src, showEmojis, showHighlight, highlightStyle = 'golden', captionFont = 'impact' }) {
  const fontCss = CAPTION_FONTS.find(f => f.id === captionFont)?.css || CAPTION_FONTS[0].css
  const videoRef = useRef(null)
  const rafRef   = useRef(null)
  const [t, setT] = useState(0)
  const chunks   = useMemo(() => chunkWords(src.words || [], 5), [src.words])

  useEffect(() => {
    const v = videoRef.current; if (!v) return
    const tick   = () => { setT(v.currentTime); if (!v.paused && !v.ended) rafRef.current = requestAnimationFrame(tick) }
    const onPlay  = () => { rafRef.current = requestAnimationFrame(tick) }
    const onPause = () => { cancelAnimationFrame(rafRef.current); setT(v.currentTime) }
    const onSeek  = () => setT(v.currentTime)
    v.addEventListener('play', onPlay); v.addEventListener('pause', onPause); v.addEventListener('seeked', onSeek)
    return () => { cancelAnimationFrame(rafRef.current); v.removeEventListener('play', onPlay); v.removeEventListener('pause', onPause); v.removeEventListener('seeked', onSeek) }
  }, [])

  const activeChunk = chunks.find(ch => t >= ch[0].start - 0.05 && t <= ch[ch.length-1].end + 0.25)

  return (
    <div className="clip-video-wrapper">
      <video ref={videoRef} src={src.url} poster={src.thumbnail} controls />
      {activeChunk && (
        <div className="clip-captions" style={{ fontFamily: fontCss }}>
          {activeChunk.map((w, i, arr) => {
            const next = i < arr.length - 1 ? arr[i+1].start : w.end + 0.1
            const on    = t >= w.start && t < next
            const hlCls = on && showHighlight !== false ? ` active ${HL_CLASS[highlightStyle] || 'hl-golden'}` : ''
            return <span key={i} className={`caption-word${hlCls}`}>{censorWord(w.word)}</span>
          })}
          {showEmojis && (() => { const e = pickEmoji(activeChunk); return e ? <span className="caption-emoji">{e}</span> : null })()}
        </div>
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// AVATAR
// ══════════════════════════════════════════════════════════
function Avatar({ name, size = 32 }) {
  return (
    <div className="avatar" style={{ width: size, height: size, fontSize: size * 0.38, background: avatarColor(name) }}>
      {initials(name)}
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// AUTH MODAL  (sign in · sign up · forgot password · reset)
// ══════════════════════════════════════════════════════════
function AuthModal({ onClose, onAuth }) {
  const [tab,        setTab]        = useState('signin')  // 'signin' | 'signup'
  const [forgot,     setForgot]     = useState(false)     // true = show forgot flow
  const [forgotSent, setForgotSent] = useState(false)     // true = token was dispatched
  const [name,  setName]  = useState('')
  const [email, setEmail] = useState('')
  const [pw,    setPw]    = useState('')
  const [token, setToken] = useState('')
  const [newPw, setNewPw] = useState('')
  const [confPw,setConfPw]= useState('')
  const [err,   setErr]   = useState('')
  const [msg,   setMsg]   = useState('')
  const [busy,  setBusy]  = useState(false)
  const api = makeApi(null)

  const resetForgot = () => { setForgot(false); setForgotSent(false); setToken(''); setNewPw(''); setConfPw(''); setErr(''); setMsg('') }

  const submitAuth = async (e) => {
    e.preventDefault(); setErr(''); setBusy(true)
    try {
      const data = await api(
        tab === 'signup' ? '/api/auth/register' : '/api/auth/login',
        { method: 'POST', body: JSON.stringify(tab === 'signup' ? { name, email, password: pw } : { email, password: pw }) }
      )
      onAuth(data.token, data.user)
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  const submitForgot = async (e) => {
    e.preventDefault(); setErr(''); setBusy(true)
    try {
      await api('/api/auth/forgot-password', { method: 'POST', body: JSON.stringify({ email }) })
      setForgotSent(true)
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  const submitReset = async (e) => {
    e.preventDefault(); setErr('')
    if (newPw !== confPw) { setErr('Passwords do not match'); return }
    setBusy(true)
    try {
      await api('/api/auth/reset-password', { method: 'POST', body: JSON.stringify({ token: token.trim(), new_password: newPw }) })
      setMsg('Password reset! You can now sign in.')
      setTimeout(() => { resetForgot(); setMsg('') }, 2500)
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal auth-modal" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>

        {/* ── Forgot password flow ── */}
        {forgot ? (
          <>
            <button className="auth-back-btn" onClick={resetForgot}>← Back to sign in</button>
            <h2 className="auth-forgot-title">
              {forgotSent ? (msg ? '✅ Done' : 'Enter your reset token') : 'Reset your password'}
            </h2>

            {msg && <div className="auth-success">{msg}</div>}

            {!forgotSent && (
              <form className="auth-form" onSubmit={submitForgot}>
                <p className="auth-forgot-hint">Enter your account email. A one-time reset token will be printed to the <strong>server console</strong>.</p>
                <div className="field"><label>Email</label><input type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required autoFocus /></div>
                {err && <div className="auth-error">{err}</div>}
                <button type="submit" className="auth-submit" disabled={busy}>{busy?'Sending…':'Send Reset Token'}</button>
              </form>
            )}

            {forgotSent && !msg && (
              <form className="auth-form" onSubmit={submitReset}>
                <p className="auth-forgot-hint">Check your server console for the token. It expires in <strong>15 minutes</strong>.</p>
                <div className="field"><label>Reset Token</label><input type="text" placeholder="Paste token from console" value={token} onChange={e=>setToken(e.target.value)} required autoFocus spellCheck={false} /></div>
                <div className="field"><label>New Password</label><input type="password" placeholder="At least 6 characters" value={newPw} onChange={e=>setNewPw(e.target.value)} required /></div>
                <div className="field"><label>Confirm Password</label><input type="password" value={confPw} onChange={e=>setConfPw(e.target.value)} required /></div>
                {err && <div className="auth-error">{err}</div>}
                <button type="submit" className="auth-submit" disabled={busy}>{busy?'Resetting…':'Reset Password'}</button>
              </form>
            )}
          </>
        ) : (
          /* ── Normal sign in / sign up ── */
          <>
            <div className="auth-tabs">
              <button className={`auth-tab${tab==='signin'?' active':''}`} onClick={() => { setTab('signin'); setErr('') }}>Sign In</button>
              <button className={`auth-tab${tab==='signup'?' active':''}`} onClick={() => { setTab('signup'); setErr('') }}>Sign Up</button>
            </div>
            <form className="auth-form" onSubmit={submitAuth}>
              {tab === 'signup' && <div className="field"><label>Name</label><input type="text" placeholder="Your name" value={name} onChange={e=>setName(e.target.value)} required autoFocus /></div>}
              <div className="field"><label>Email</label><input type="email" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required autoFocus={tab==='signin'} /></div>
              <div className="field">
                <div className="field-label-row">
                  <label>Password</label>
                  {tab === 'signin' && <button type="button" className="forgot-link" onClick={() => { setForgot(true); setErr('') }}>Forgot password?</button>}
                </div>
                <input type="password" placeholder={tab==='signup'?'At least 6 characters':'Your password'} value={pw} onChange={e=>setPw(e.target.value)} required />
              </div>
              {err && <div className="auth-error">{err}</div>}
              <button type="submit" className="auth-submit" disabled={busy}>{busy ? 'Please wait…' : tab==='signup' ? 'Create Account' : 'Sign In'}</button>
            </form>
            <p className="auth-switch">
              {tab==='signin' ? "Don't have an account? " : 'Already have an account? '}
              <button onClick={() => { setTab(tab==='signin'?'signup':'signin'); setErr('') }}>{tab==='signin'?'Sign Up':'Sign In'}</button>
            </p>
          </>
        )}
      </div>
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// PROFILE MODAL
// ══════════════════════════════════════════════════════════
function ProfileModal({ user, token, onClose }) {
  const [stats, setStats] = useState(null)
  useEffect(() => { makeApi(token)('/api/auth/me').then(d => setStats(d.stats)).catch(() => {}) }, [token])
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal profile-modal" onClick={e=>e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        <div className="profile-avatar-wrap"><Avatar name={user.name} size={72} /></div>
        <h2 className="profile-name">{user.name}</h2>
        <p className="profile-email">{user.email}</p>
        <p className="profile-since">Member since {fmtDate(user.created_at)}</p>
        {stats && (
          <div className="profile-stats">
            <div className="profile-stat"><span className="stat-value">{stats.jobs}</span><span className="stat-label">Videos Processed</span></div>
            <div className="profile-stat-divider" />
            <div className="profile-stat"><span className="stat-value">{stats.clips}</span><span className="stat-label">Clips Generated</span></div>
          </div>
        )}
      </div>
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// SETTINGS MODAL
// ══════════════════════════════════════════════════════════
function SettingsModal({ user, token, onClose, onUserUpdate }) {
  const [tab,setTab]     = useState('profile')
  const [name,setName]   = useState(user.name)
  const [email,setEmail] = useState(user.email)
  const [da,setDa]       = useState(user.default_annotate)
  const [dh,setDh]       = useState(user.default_highlight)
  const [de,setDe]       = useState(user.default_emojis)
  const [cp,setCp]       = useState(''); const [np,setNp]=useState(''); const [cf,setCf]=useState('')
  const [busy,setBusy]   = useState(false)
  const [msg,setMsg]     = useState(''); const [err,setErr]=useState('')

  const api   = makeApi(token)
  const flash = (ok, txt) => { if(ok) setMsg(txt); else setErr(txt); setTimeout(()=>{setMsg('');setErr('')},3000) }

  const saveProfile = async (e) => {
    e.preventDefault(); setBusy(true); setErr(''); setMsg('')
    try { const u = await api('/api/auth/me',{method:'PUT',body:JSON.stringify({name,email})}); onUserUpdate(u); flash(true,'Profile updated!') }
    catch(ex){flash(false,ex.message)} finally{setBusy(false)}
  }
  const savePrefs = async () => {
    setBusy(true); setErr(''); setMsg('')
    try { const u = await api('/api/auth/me',{method:'PUT',body:JSON.stringify({default_annotate:da,default_highlight:dh,default_emojis:de})}); onUserUpdate(u); flash(true,'Preferences saved!') }
    catch(ex){flash(false,ex.message)} finally{setBusy(false)}
  }
  const savePw = async (e) => {
    e.preventDefault()
    if(np!==cf){flash(false,'New passwords do not match');return}
    setBusy(true); setErr(''); setMsg('')
    try { await api('/api/auth/password',{method:'PUT',body:JSON.stringify({current_password:cp,new_password:np})}); setCp('');setNp('');setCf(''); flash(true,'Password changed!') }
    catch(ex){flash(false,ex.message)} finally{setBusy(false)}
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal settings-modal" onClick={e=>e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        <h2 className="modal-title">Settings</h2>
        <div className="settings-tabs">
          {['profile','preferences','password'].map(t=>(
            <button key={t} className={`settings-tab${tab===t?' active':''}`} onClick={()=>{setTab(t);setErr('');setMsg('')}}>
              {t.charAt(0).toUpperCase()+t.slice(1)}
            </button>
          ))}
        </div>
        {msg && <div className="settings-msg success">{msg}</div>}
        {err && <div className="settings-msg error">{err}</div>}

        {tab==='profile' && (
          <form className="auth-form" onSubmit={saveProfile}>
            <div className="settings-avatar-row"><Avatar name={user.name} size={52} /><div><div className="settings-avatar-name">{user.name}</div><div className="settings-avatar-email">{user.email}</div></div></div>
            <div className="field"><label>Name</label><input type="text" value={name} onChange={e=>setName(e.target.value)} required /></div>
            <div className="field"><label>Email</label><input type="email" value={email} onChange={e=>setEmail(e.target.value)} required /></div>
            <button type="submit" className="auth-submit" disabled={busy}>{busy?'Saving…':'Save Changes'}</button>
          </form>
        )}

        {tab==='preferences' && (
          <div className="prefs-section">
            <p className="prefs-hint">These become your defaults each time you open the app.</p>
            <div className="pref-row"><div className="pref-label">Auto Captions<span className="pref-sub">Overlay captions on each clip</span></div><div className={`toggle${da?' on':''}`} onClick={()=>setDa(v=>!v)} /></div>
            {da && <><div className="pref-row pref-indent"><div className="pref-label">Highlight Active Word<span className="pref-sub">Yellow karaoke highlight</span></div><div className={`toggle${dh?' on':''}`} onClick={()=>setDh(v=>!v)} /></div>
            <div className="pref-row pref-indent"><div className="pref-label">Emojis<span className="pref-sub">Context-aware emoji per chunk</span></div><div className={`toggle${de?' on':''}`} onClick={()=>setDe(v=>!v)} /></div></>}
            <button className="auth-submit" style={{marginTop:24}} onClick={savePrefs} disabled={busy}>{busy?'Saving…':'Save Preferences'}</button>
          </div>
        )}

        {tab==='password' && (
          <form className="auth-form" onSubmit={savePw}>
            <div className="field"><label>Current Password</label><input type="password" value={cp} onChange={e=>setCp(e.target.value)} required /></div>
            <div className="field"><label>New Password</label><input type="password" placeholder="At least 6 characters" value={np} onChange={e=>setNp(e.target.value)} required /></div>
            <div className="field"><label>Confirm New Password</label><input type="password" value={cf} onChange={e=>setCf(e.target.value)} required /></div>
            <button type="submit" className="auth-submit" disabled={busy}>{busy?'Saving…':'Change Password'}</button>
          </form>
        )}
      </div>
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// HISTORY VIEW
// ══════════════════════════════════════════════════════════
function HistoryView({ token }) {
  const [items,    setItems]    = useState([])
  const [loading,  setLoading]  = useState(true)
  const [expanded, setExpanded] = useState({})

  useEffect(() => {
    makeApi(token)('/api/history').then(d=>setItems(d.history||[])).catch(()=>{}).finally(()=>setLoading(false))
  }, [token])

  const remove = async (id) => { try { await makeApi(token)(`/api/history/${id}`,{method:'DELETE'}); setItems(p=>p.filter(i=>i.id!==id)) } catch {} }
  const toggle = (id) => setExpanded(p=>({...p,[id]:!p[id]}))
  const fmt = (s) => { const m=Math.floor(s/60); return `${m}:${Math.floor(s%60).toString().padStart(2,'0')}` }

  if (loading) return <div className="history-empty">Loading history…</div>
  if (!items.length) return <div className="history-empty"><div className="history-empty-icon">📂</div><p>No history yet. Process a video to get started.</p></div>

  return (
    <div className="history-list">
      {items.map(item=>(
        <div key={item.id} className="history-item">
          <div className="history-item-header">
            <div className="history-item-meta">
              <span className={`history-mode-badge ${item.mode}`}>{item.mode==='annotate'?'Annotate':'Find Clips'}</span>
              <span className="history-filename">{item.filename||'Untitled'}</span>
            </div>
            <div className="history-item-right">
              <span className="history-date">{fmtDate(item.created_at)}</span>
              <span className="history-clip-count">{item.clips_count} clip{item.clips_count!==1?'s':''}</span>
              {item.clips_count>0 && <button className="history-expand-btn" onClick={()=>toggle(item.id)}>{expanded[item.id]?'Hide':'View'}</button>}
              <button className="history-delete-btn" onClick={()=>remove(item.id)} title="Delete">✕</button>
            </div>
          </div>

          {item.clips_count>0 && !expanded[item.id] && item.clips && (
            <div className="history-thumbs">
              {item.clips.slice(0,5).map((c,i)=>
                c.thumbnail
                  ? <img key={i} src={c.thumbnail} className="history-thumb" alt={c.title} />
                  : <div key={i} className="history-thumb history-thumb-placeholder">🎬</div>
              )}
              {item.clips.length>5 && <div className="history-thumb history-thumb-more">+{item.clips.length-5}</div>}
            </div>
          )}

          {expanded[item.id] && item.clips && (
            <div className="history-clips-grid">
              {item.clips.map((clip,i)=>(
                <div key={i} className="clip">
                  {clip.url?.endsWith('.mp4')
                    ? clip.words?.length ? <CaptionedVideo src={clip} showEmojis={false} showHighlight={true} /> : <video src={clip.url} poster={clip.thumbnail} controls />
                    : <audio src={clip.url} controls />}
                  <div className="clip-header"><h3>{clip.title||`Clip ${i+1}`}</h3>{clip.score!=null&&<span className="badge">⚡ {clip.score}</span>}</div>
                  {clip.start!=null&&<div className="clip-timestamp">{fmt(clip.start)} &ndash; {fmt(clip.end)}</div>}
                  {clip.text&&<p className="clip-transcript">"{censorText(clip.text)}"</p>}
                  <a className="clip-download" href={clip.url} download>↓ Download Clip</a>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// ABOUT PAGE
// ══════════════════════════════════════════════════════════
const FEATURES = [
  { icon: '🎯', title: 'Smart Clip Detection',    desc: 'Multi-signal AI scores every second of your video — sentiment, keywords, audio energy, hooks, and more.' },
  { icon: '📝', title: 'Karaoke Captions',        desc: 'Word-by-word captions with a live yellow highlight that follows along as the speaker talks.' },
  { icon: '🖼️', title: 'Auto Thumbnails',         desc: 'Beautiful 9:16 thumbnails with blurred background compositing and a burned-in title overlay.' },
  { icon: '🎬', title: 'Annotate Mode',            desc: 'Upload any clip and get instant captioned output — no cutting, just pure annotation.' },
  { icon: '📊', title: 'Engagement Scoring',      desc: 'Every clip gets a virality score based on emotional intensity, pace, hooks, and profanity.' },
  { icon: '💾', title: 'History & Accounts',      desc: 'Sign in to save every processing job. Revisit, re-download, or delete past clips any time.' },
]

const TECH = [
  { emoji: '🎙️', name: 'faster-whisper' },
  { emoji: '🧠', name: 'VADER Sentiment' },
  { emoji: '🎞️', name: 'FFmpeg' },
  { emoji: '⚡', name: 'FastAPI' },
  { emoji: '⚛️', name: 'React' },
]

const CREATORS = ['Podcasters','Streamers','YouTubers','Interview Shows','Stand-up Comedians','Educators','Sports Creators','Talk Show Hosts']

function AboutPage({ onGetStarted }) {
  return (
    <div className="about-page">

      {/* ── Hero ── */}
      <div className="about-hero fade-in">
        <div className="about-hero-bg" />
        <div className="about-eyebrow">✦ Open Source · AI-Powered</div>
        <h1>The AI engine behind<br />your next viral clip.</h1>
        <p>Drop a long video. Walk away with short-form gold. Snipflow finds the moments worth sharing — automatically.</p>
        <div className="about-ctas">
          <button className="btn-primary" onClick={onGetStarted}>Try it for free</button>
          <a className="btn-ghost" href="#how-it-works">How it works ↓</a>
        </div>
      </div>

      {/* ── How it works ── */}
      <div className="about-section" id="how-it-works">
        <div className="section-label">Process</div>
        <h2 className="section-title">Three steps.<br />Infinite clips.</h2>
        <p className="section-sub">No editing experience required. No timeline dragging. No guessing.</p>
        <div className="steps-grid">
          {[
            { num: '01', icon: '📤', title: 'Upload', desc: 'Drop any long-form video — podcast, stream, interview, vlog. MP4, MOV, AVI all work.' },
            { num: '02', icon: '🤖', title: 'Analyze', desc: 'The AI scores every segment using 12+ signals: sentiment, keyword hooks, vocal energy, pacing, and more.' },
            { num: '03', icon: '✂️', title: 'Export', desc: 'Get short clips, auto-thumbnails, and karaoke captions. Download and post.' },
          ].map(s => (
            <div key={s.num} className="step">
              <div className="step-num">{s.num}</div>
              <span className="step-icon">{s.icon}</span>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Features ── */}
      <div className="about-section">
        <div className="section-label">Features</div>
        <h2 className="section-title">Everything you need.<br />Nothing you don't.</h2>
        <p className="section-sub">Snipflow is opinionated: it does a small number of things exceptionally well.</p>
        <div className="features-grid">
          {FEATURES.map(f => (
            <div key={f.title} className="feature-card">
              <div className="feature-icon">{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Built for ── */}
      <div className="about-section">
        <div className="section-label">Audience</div>
        <h2 className="section-title">Built for creators<br />who move fast.</h2>
        <p className="section-sub">If you produce long-form content and want short-form reach, Snipflow was made for you.</p>
        <div className="creator-row">
          {CREATORS.map(c => <div key={c} className="creator-pill">{c}</div>)}
        </div>
      </div>

      {/* ── Tech stack ── */}
      <div className="about-section">
        <div className="section-label">Under the hood</div>
        <h2 className="section-title">Serious tech.<br />Surprisingly fast.</h2>
        <p className="section-sub">No cloud APIs, no latency tax. Everything runs locally on your machine.</p>
        <div className="tech-row">
          {TECH.map(t => (
            <div key={t.name} className="tech-pill"><span>{t.emoji}</span>{t.name}</div>
          ))}
        </div>
      </div>

      {/* ── CTA Banner ── */}
      <div className="about-cta-banner">
        <h2>Ready to go viral?</h2>
        <p>Upload your first video in under 30 seconds.<br />No account required to try.</p>
        <button className="btn-primary" onClick={onGetStarted}>Start Snipflowing →</button>
      </div>

    </div>
  )
}


// ══════════════════════════════════════════════════════════
// ERROR PAGES
// ══════════════════════════════════════════════════════════
function NotFoundPage({ onHome }) {
  return (
    <div className="error-page">
      <div className="error-404-num" data-text="404">404</div>
      <h1 className="error-title">This page didn't make the cut.</h1>
      <p className="error-subtitle">
        Our AI reviewed this URL and ran a full engagement analysis.
        The results were not good.
      </p>

      <div className="error-analysis">
        <div className="error-analysis-title">AI Clip Analysis Report</div>
        <div className="ai-row"><span className="ai-key">URL Virality Score</span>     <span className="ai-val bad">0 / 100</span></div>
        <div className="ai-row"><span className="ai-key">Sentiment (VADER)</span>      <span className="ai-val meh">😐  0.00 (flat)</span></div>
        <div className="ai-row"><span className="ai-key">Engagement Signals</span>     <span className="ai-val none">none detected</span></div>
        <div className="ai-row"><span className="ai-key">Hook Count</span>             <span className="ai-val bad">0</span></div>
        <div className="ai-row"><span className="ai-key">Emotional Peaks</span>        <span className="ai-val bad">0</span></div>
        <div className="ai-row"><span className="ai-key">Recommendation</span>         <span className="ai-val bad">CLIP DISCARDED ✕</span></div>
      </div>

      <div className="error-progress-wrap">
        <div className="error-progress-label"><span>Processing URL…</span><span style={{color:'var(--danger)'}}>Stuck</span></div>
        <div className="error-progress-track"><div className="error-progress-bar" /></div>
      </div>

      <div className="error-btns" style={{ marginTop: 28 }}>
        <button className="error-home-btn" onClick={onHome}>← Back to Snipflow</button>
      </div>
    </div>
  )
}

function CrashPage({ error, onRetry, onHome }) {
  const [showDetails, setShowDetails] = useState(false)
  return (
    <div className="error-page">
      <span className="error-500-icon">🤖</span>
      <h1 className="error-title">Pipeline crash detected.</h1>
      <p className="error-subtitle">
        Something exploded in our AI pipeline. The model is currently transcribing the error message,
        scoring it for emotional impact <strong style={{color:'var(--warning)'}}>( 9.1 / 10 — very distressing )</strong>,
        and extracting it as a highlight clip.
      </p>

      <div className="error-terminal">
        <div className="terminal-header">
          <div className="terminal-dot red" />
          <div className="terminal-dot yellow" />
          <div className="terminal-dot green" />
        </div>
        <div className="terminal-line">$ snipflow-pipeline <span className="t-key">--mode</span> process</div>
        <div className="terminal-line"><span className="t-key">Loading</span> Whisper base model... <span className="t-ok">done</span></div>
        <div className="terminal-line"><span className="t-key">Transcribing</span> audio stream...</div>
        <div className="terminal-line"><span className="t-err">✗ FATAL:</span> {error?.message || 'unexpected internal error'}</div>
        <div className="terminal-line"><span className="t-key">Sentiment score</span> of error: <span className="t-err">-1.0 (catastrophic)</span></div>
        <div className="terminal-line"><span className="t-key">Filing</span> emotional support request... <span className="t-ok">submitted</span></div>
        <div className="terminal-line"><span className="t-cursor" /></div>
      </div>

      {showDetails && error?.stack && (
        <pre style={{ fontSize: 11, color: 'var(--text-6)', maxWidth: 520, textAlign: 'left', marginBottom: 24, whiteSpace: 'pre-wrap', lineHeight: 1.6, background: 'var(--surface)', padding: 16, borderRadius: 'var(--r-md)', border: '1px solid var(--border)' }}>
          {error.stack}
        </pre>
      )}

      <div style={{ marginBottom: 24 }}>
        <button onClick={() => setShowDetails(v => !v)} style={{ background: 'none', border: 'none', color: 'var(--text-6)', fontSize: 12, cursor: 'pointer', textDecoration: 'underline', textUnderlineOffset: 2 }}>
          {showDetails ? 'Hide' : 'Show'} stack trace
        </button>
      </div>

      <div className="error-btns">
        <button className="error-home-btn" onClick={onHome}>← Go Home</button>
        <button className="error-retry-btn" onClick={onRetry}>Restart Pipeline</button>
      </div>
    </div>
  )
}

// React error boundary (must be class component)
class ErrorBoundary extends Component {
  state = { error: null }
  static getDerivedStateFromError(error) { return { error } }
  componentDidCatch(error, info) { console.error('ErrorBoundary caught:', error, info) }
  render() {
    if (this.state.error) {
      return (
        <CrashPage
          error={this.state.error}
          onRetry={() => this.setState({ error: null })}
          onHome={() => { this.setState({ error: null }); window.history.pushState({}, '', '/') }}
        />
      )
    }
    return this.props.children
  }
}


// ══════════════════════════════════════════════════════════
// CLIP COUNT PICKER
// ══════════════════════════════════════════════════════════
const COUNT_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

function ClipCountPicker({ value, onChange }) {
  return (
    <div className="count-picker-wrap">
      <div className="length-picker-label">Number of Clips</div>
      <div className="count-picker">
        {COUNT_OPTIONS.map(n => (
          <button
            key={n}
            className={`count-option${value === n ? ' active' : ''}`}
            onClick={() => onChange(n)}
          >
            {n === 0 ? 'Auto' : n}
          </button>
        ))}
      </div>
      <div className="length-hint">
        <span className="length-hint-text">
          {value === 0 ? 'AI picks the best clips automatically based on engagement scores' : `Exactly ${value} clip${value !== 1 ? 's' : ''} will be returned, sorted by score`}
        </span>
      </div>
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// HIGHLIGHT STYLE PICKER
// ══════════════════════════════════════════════════════════
const HL_STYLES = [
  { id: 'golden',    label: 'Golden'    },
  { id: 'neon',      label: 'Neon'      },
  { id: 'flamingo',  label: 'Flamingo'  },
  { id: 'electric',  label: 'Electric'  },
  { id: 'ember',     label: 'Ember'     },
  { id: 'violet',    label: 'Violet'    },
  { id: 'rainbow',   label: 'Rainbow'   },
  { id: 'subtitles', label: 'Subtitles' },
]

function HighlightStylePicker({ value, onChange }) {
  return (
    <div className="hl-picker-wrap">
      <div className="length-picker-label">Highlight Style</div>
      <div className="hl-picker">
        {HL_STYLES.map(s => (
          <button
            key={s.id}
            className={`hl-option${value === s.id ? ' active' : ''}`}
            onClick={() => onChange(s.id)}
            title={s.label}
          >
            <span className={`hl-swatch hl-swatch-${s.id}`} />
            <span className="hl-label">{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// FONT PICKER
// ══════════════════════════════════════════════════════════
function FontPicker({ value, onChange }) {
  return (
    <div className="font-picker-wrap">
      <div className="length-picker-label">Caption Font</div>
      <div className="font-picker">
        {CAPTION_FONTS.map(f => (
          <button
            key={f.id}
            className={`font-option${value === f.id ? ' active' : ''}`}
            style={{ fontFamily: f.css }}
            onClick={() => onChange(f.id)}
          >
            <span className="font-preview">Aa</span>
            <span className="font-label">{f.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// CLIP LENGTH PICKER
// ══════════════════════════════════════════════════════════
const LENGTH_OPTIONS = [
  {
    id:      'short',
    label:   'Short',
    range:   '15 – 35s',
    hint:    'Targeting ~25s clips — snappy, scroll-stopping content',
    tags:    'TikTok · Reels',
    icon:    '⚡',
  },
  {
    id:      'medium',
    label:   'Medium',
    range:   '25 – 70s',
    hint:    'Targeting ~50s clips — the sweet spot for most platforms',
    tags:    'YouTube Shorts · Reels',
    icon:    '🎯',
  },
  {
    id:      'long',
    label:   'Long',
    range:   '60 – 90s',
    hint:    'Targeting ~75s clips — room for full stories and context',
    tags:    'YouTube Shorts · Podcasts',
    icon:    '📺',
  },
]

function ClipLengthPicker({ value, onChange }) {
  const selected = LENGTH_OPTIONS.find(o => o.id === value)
  return (
    <div className="length-picker-wrap">
      <div className="length-picker-label">Clip Length</div>
      <div className="length-picker">
        {LENGTH_OPTIONS.map(opt => (
          <button
            key={opt.id}
            className={`length-option${value === opt.id ? ' active' : ''}`}
            onClick={() => onChange(opt.id)}
          >
            <span className="length-icon">{opt.icon}</span>
            <span className="length-name">{opt.label}</span>
            <span className="length-range">{opt.range}</span>
          </button>
        ))}
      </div>
      {selected && (
        <div className="length-hint">
          <span className="length-hint-text">{selected.hint}</span>
          <span className="length-hint-tags">{selected.tags}</span>
        </div>
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════════
// APP
// ══════════════════════════════════════════════════════════
export default function App() {
  // ── Routing ─────────────────────────────────────────────
  const [view, setViewState] = useState(getViewFromPath)

  const navigate = useCallback((v) => {
    window.history.pushState({}, '', PATH_MAP[v] || '/')
    setViewState(v)
  }, [])

  useEffect(() => {
    const handler = () => setViewState(getViewFromPath())
    window.addEventListener('popstate', handler)
    return () => window.removeEventListener('popstate', handler)
  }, [])

  // ── Auth ─────────────────────────────────────────────────
  const [token, setToken] = useState(() => localStorage.getItem('clipify_token') || '')
  const [user,  setUser]  = useState(null)
  const [showAuth,     setShowAuth]     = useState(false)
  const [showProfile,  setShowProfile]  = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)
  const dropdownRef = useRef(null)

  useEffect(() => {
    if (!token) { setUser(null); return }
    makeApi(token)('/api/auth/me')
      .then(d => { setUser(d); setAnnotate(d.default_annotate); setHighlight(d.default_highlight); setEmojis(d.default_emojis) })
      .catch(() => { localStorage.removeItem('clipify_token'); setToken(''); setUser(null) })
  }, [token])

  useEffect(() => {
    if (!showDropdown) return
    const h = (e) => { if (dropdownRef.current && !dropdownRef.current.contains(e.target)) setShowDropdown(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [showDropdown])

  const handleAuth = (tok, usr) => {
    localStorage.setItem('clipify_token', tok); setToken(tok); setUser(usr)
    setAnnotate(usr.default_annotate); setHighlight(usr.default_highlight); setEmojis(usr.default_emojis)
    setShowAuth(false)
  }
  const handleSignOut = () => {
    localStorage.removeItem('clipify_token'); setToken(''); setUser(null)
    setShowDropdown(false); navigate('home'); setClips([])
  }

  // ── Process state ────────────────────────────────────────
  const [file,      setFile]      = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [progress,  setProgress]  = useState(0)
  const [statusIdx, setStatusIdx] = useState(0)
  const [mode,            setMode]            = useState('clips')
  const [clipLength,      setClipLength]      = useState('medium')
  const [maxClips,        setMaxClips]        = useState(0)
  const [highlightStyle,  setHighlightStyle]  = useState('golden')
  const [captionFont,     setCaptionFont]     = useState('impact')
  const [showMoreOptions, setShowMoreOptions] = useState(false)
  const [annotate,        setAnnotate]        = useState(false)
  const [highlight,       setHighlight]       = useState(true)
  const [emojis,          setEmojis]          = useState(false)
  const [punchyHeader,    setPunchyHeader]    = useState(true)
  const [customHook,      setCustomHook]      = useState('')
  const [clips,     setClips]     = useState([])
  const [dragOver,  setDragOver]  = useState(false)
  const fileRef   = useRef(null)
  const progRef   = useRef(null)
  const statRef   = useRef(null)
  const abortRef  = useRef(null)

  const activeStatus = mode === 'annotate' ? ANNOTATE_STATUS : CLIP_STATUS

  useEffect(() => {
    if (loading) {
      setProgress(0); setStatusIdx(0)
      progRef.current = setInterval(() => setProgress(p => p + (95-p)*0.025), 600)
      statRef.current = setInterval(() => setStatusIdx(i => Math.min(i+1, activeStatus.length-1)), 18000)
    } else {
      clearInterval(progRef.current); clearInterval(statRef.current)
    }
    return () => { clearInterval(progRef.current); clearInterval(statRef.current) }
  }, [loading])

  const handleFile     = (f) => { if (f && f.type.startsWith('video/')) setFile(f) }
  const handleDrop     = useCallback((e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }, [])
  const handleDragOver = useCallback((e) => { e.preventDefault(); setDragOver(true) }, [])
  const handleDragLeave= useCallback(() => setDragOver(false), [])

  const handleUpload = async () => {
    if (!file) return
    setLoading(true); setClips([])
    const ctrl = new AbortController(); abortRef.current = ctrl
    const tid  = setTimeout(() => ctrl.abort(), 30*60*1000)
    try {
      const fd = new FormData(); fd.append('file', file)
      if (mode==='annotate') {
        fd.append('annotate_only', true)
      } else {
        fd.append('annotate', annotate)
        fd.append('clip_length', clipLength)
        fd.append('max_clips', maxClips)
      }
      fd.append('header_enabled', punchyHeader)
      if (customHook.trim()) fd.append('custom_hook', customHook.trim())
      const headers = {}; if (token) headers['Authorization'] = `Bearer ${token}`
      const res = await fetch('/api/process', { method:'POST', body:fd, signal:ctrl.signal, headers })
      if (!res.ok) { const e=await res.json().catch(()=>({})); throw new Error(e.detail||`Server error ${res.status}`) }
      const data = await res.json(); setProgress(100)
      await new Promise(r=>setTimeout(r,400)); setClips(data.clips||[])
    } catch(err) {
      if (err.name==='AbortError') alert('Processing timed out. Try a shorter video.')
      else { console.error(err); alert(err.message||'Something went wrong.') }
      setProgress(0)
    } finally { clearTimeout(tid); setLoading(false) }
  }

  const fmt = (s) => { const m=Math.floor(s/60); return `${m}:${Math.floor(s%60).toString().padStart(2,'0')}` }

  // ── Render ───────────────────────────────────────────────
  return (
    <ErrorBoundary>
      {/* ── Shared Nav ── */}
      <div className="page">
        <nav className="nav">
          <div className="nav-left">
            <span className="nav-logo" onClick={() => navigate('home')} style={{cursor:'pointer'}}>Snipflow</span>
            <button className={`nav-link${view==='about'?' active':''}`} onClick={() => navigate('about')}>About</button>
          </div>
          {user ? (
            <div className="user-chip-wrap" ref={dropdownRef}>
              <button className="user-chip" onClick={() => setShowDropdown(v=>!v)}>
                <Avatar name={user.name} size={26} />
                <span>{user.name.split(' ')[0]}</span>
                <span className="user-chip-caret">{showDropdown?'▲':'▼'}</span>
              </button>
              {showDropdown && (
                <div className="user-dropdown">
                  <button onClick={() => { setShowDropdown(false); setShowProfile(true) }}>My Profile</button>
                  <button onClick={() => { setShowDropdown(false); navigate('history') }}>History</button>
                  <button onClick={() => { setShowDropdown(false); setShowSettings(true) }}>Settings</button>
                  <div className="dropdown-divider" />
                  <button className="dropdown-signout" onClick={handleSignOut}>Sign Out</button>
                </div>
              )}
            </div>
          ) : (
            <button className="nav-btn" onClick={() => setShowAuth(true)}>Sign In</button>
          )}
        </nav>

        {/* ── Modals ── */}
        {showAuth    && <AuthModal onClose={() => setShowAuth(false)} onAuth={handleAuth} />}
        {showProfile && user && <ProfileModal user={user} token={token} onClose={() => setShowProfile(false)} />}
        {showSettings && user && (
          <SettingsModal user={user} token={token} onClose={() => setShowSettings(false)} onUserUpdate={u => setUser(prev => ({...prev,...u}))} />
        )}

        {/* ══ VIEW: 404 ══ */}
        {view === '404' && <NotFoundPage onHome={() => navigate('home')} />}

        {/* ══ VIEW: ABOUT ══ */}
        {view === 'about' && <AboutPage onGetStarted={() => navigate('home')} />}

        {/* ══ VIEW: HISTORY ══ */}
        {view === 'history' && (
          <>
            <div className="results-header" style={{ marginBottom: 28 }}>
              <h2>History</h2>
              <button className="nav-btn" style={{ fontSize: 12 }} onClick={() => navigate('home')}>← Back</button>
            </div>
            <HistoryView token={token} />
          </>
        )}

        {/* ══ VIEW: HOME ══ */}
        {view === 'home' && (
          <>
            <section className="hero">
              <div className="hero-badge">✦ AI-Powered</div>
              <h1>Turn Long Videos<br />Into Viral Clips</h1>
              <p>Upload any video and let AI find the most engaging moments — instantly.</p>
            </section>

            <div className="upload-wrapper">
              <div className="mode-selector">
                <button className={`mode-btn${mode==='clips'?' active':''}`} onClick={() => { setMode('clips'); setClips([]) }}>Find Clips</button>
                <button className={`mode-btn${mode==='annotate'?' active':''}`} onClick={() => { setMode('annotate'); setClips([]) }}>Annotate Video</button>
              </div>

              {mode === 'clips' && <ClipLengthPicker value={clipLength} onChange={setClipLength} />}
              {mode === 'clips' && <ClipCountPicker value={maxClips} onChange={setMaxClips} />}

              <div
                className={`upload-card${dragOver?' drag-over':''}`}
                onClick={() => fileRef.current?.click()}
                onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
              >
                <input ref={fileRef} type="file" accept="video/*" style={{display:'none'}} onChange={e=>handleFile(e.target.files[0])} />
                <div className="upload-icon">🎬</div>
                {file
                  ? <div className="upload-file-name">📎 {file.name}</div>
                  : <><div className="upload-title">Drop your video here</div><div className="upload-subtitle">MP4, MOV, AVI &mdash; click or drag to upload</div></>
                }
              </div>

              {mode === 'clips' && (
                <label className="annotate-toggle">
                  <input type="checkbox" checked={annotate} onChange={e => { setAnnotate(e.target.checked); if (!e.target.checked) { setEmojis(false); setHighlight(true) } }} />
                  <span className="annotate-label">
                    Auto Captions
                    <span className="annotate-hint">overlay captions on each clip</span>
                  </span>
                  {annotate && (
                    <div className="caption-sub-toggles" onClick={e => e.stopPropagation()}>
                      <label className="emoji-toggle"><input type="checkbox" checked={highlight} onChange={e=>setHighlight(e.target.checked)} /><span>✦ Highlight</span></label>
                      <label className="emoji-toggle"><input type="checkbox" checked={emojis} onChange={e=>setEmojis(e.target.checked)} /><span>✨ Emojis</span></label>
                    </div>
                  )}
              </label>
              )}
              {mode === 'clips' && annotate && highlight && (
                <HighlightStylePicker value={highlightStyle} onChange={setHighlightStyle} />
              )}

              {(mode === 'annotate' || (mode === 'clips' && annotate)) && (
                <div className="more-options-wrap">
                  <button className="more-options-toggle" onClick={() => setShowMoreOptions(v => !v)}>
                    <span>✦ Caption Options</span>
                    <span className="more-options-caret">{showMoreOptions ? '▲' : '▼'}</span>
                  </button>
                  {showMoreOptions && (
                    <div className="more-options-panel">
                      <FontPicker value={captionFont} onChange={setCaptionFont} />
                      <label className="more-option-row">
                        <input type="checkbox" checked={punchyHeader} onChange={e => setPunchyHeader(e.target.checked)} />
                        <span className="more-option-label">
                          Add Punchy Header
                          <span className="more-option-hint">burn hook title at clip start</span>
                        </span>
                      </label>
                      <div className="more-option-input-wrap">
                        <label className="more-option-label" htmlFor="custom-hook-input">
                          Custom Hook
                          <span className="more-option-hint">override the generated title</span>
                        </label>
                        <input
                          id="custom-hook-input"
                          className="custom-hook-input"
                          type="text"
                          placeholder="e.g. HE REALLY SAID THIS 😳"
                          maxLength={120}
                          value={customHook}
                          onChange={e => setCustomHook(e.target.value)}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              <button className="generate-btn" onClick={handleUpload} disabled={!file || loading}>
                {loading ? 'Processing…' : mode==='annotate' ? 'Annotate Video' : 'Generate Clips'}
              </button>

              {!user && (
                <p className="signin-nudge">
                  <button onClick={() => setShowAuth(true)}>Sign in</button> to save your processing history
                </p>
              )}
            </div>

            {loading && (
              <div className="loading">
                <p>{activeStatus[statusIdx]}</p>
                <div className="progress-track"><div className="progress-bar" style={{width:`${progress}%`}} /></div>
                <div className="loading-footer">
                  <span className="progress-label">{Math.round(progress)}%</span>
                  <button className="cancel-btn" onClick={() => abortRef.current?.abort()}>Cancel</button>
                </div>
              </div>
            )}

            {!loading && clips.length > 0 && (
              <section>
                <div className="results-header">
                  <h2>{mode==='annotate' ? 'Annotated Video' : 'Your Clips'}</h2>
                  {mode==='clips' && <span className="results-count">{clips.length} clip{clips.length!==1?'s':''} found</span>}
                </div>
                <div className="clip-grid">
                  {clips.map((clip, idx) => (
                    <div key={idx} className="clip">
                      {clip.url?.endsWith('.mp4')
                        ? clip.words?.length ? <CaptionedVideo src={clip} showEmojis={emojis} showHighlight={mode==='annotate'?true:highlight} highlightStyle={highlightStyle} captionFont={captionFont} /> : <video src={clip.url} poster={clip.thumbnail} controls />
                        : <audio src={clip.url} controls />}
                      <div className="clip-header"><h3>{clip.title||`Clip ${idx+1}`}</h3>{clip.score!=null&&<span className="badge">⚡ {clip.score}</span>}</div>
                      <div className="clip-timestamp">{fmt(clip.start)} &ndash; {fmt(clip.end)}</div>
                      {clip.summary && (
                        <div className="clip-summary-block">
                          <div className="clip-summary-label">AI Summary</div>
                          <p className="clip-summary">{censorText(clip.summary)}</p>
                        </div>
                      )}
                      <a className="clip-download" href={clip.url} download>↓ Download Clip</a>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </ErrorBoundary>
  )
}
