import { useState, useRef, useCallback } from 'react'
import './styles.css'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [clips, setClips] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = (f) => {
    if (f && f.type.startsWith('video/')) setFile(f)
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => setDragOver(false), [])

  const handleUpload = async () => {
    if (!file) return

    setLoading(true)
    setClips([])

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Processing failed')
      }

      const data = await res.json()
      setClips(data.clips || [])
    } catch (err) {
      console.error(err)
      alert(err.message || 'Something went wrong processing the video')
    }

    setLoading(false)
  }

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="page">
      <nav className="nav">
        <span className="nav-logo">Clipify</span>
        <button className="nav-btn">Sign In</button>
      </nav>

      <section className="hero">
        <div className="hero-badge">✦ AI-Powered</div>
        <h1>Turn Long Videos<br />Into Viral Clips</h1>
        <p>Upload any video and let AI find the most engaging moments — instantly.</p>
      </section>

      <div className="upload-wrapper">
        <div
          className={`upload-card${dragOver ? ' drag-over' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className="upload-icon">🎬</div>
          {file ? (
            <div className="upload-file-name">📎 {file.name}</div>
          ) : (
            <>
              <div className="upload-title">Drop your video here</div>
              <div className="upload-subtitle">MP4, MOV, AVI &mdash; click or drag to upload</div>
            </>
          )}
        </div>

        <button
          className="generate-btn"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? 'Processing...' : 'Generate Clips'}
        </button>
      </div>

      {loading && (
        <div className="loading">
          <div className="loading-spinner" />
          <p>Analyzing your video&hellip;</p>
        </div>
      )}

      {!loading && clips.length > 0 && (
        <section>
          <div className="results-header">
            <h2>Your Clips</h2>
            <span className="results-count">
              {clips.length} clip{clips.length !== 1 ? 's' : ''} found
            </span>
          </div>
          <div className="clip-grid">
            {clips.map((clip, index) => (
              <div key={index} className="clip">
                {clip.url?.endsWith('.mp4') ? (
                  <video src={clip.url} controls />
                ) : (
                  <audio src={clip.url} controls />
                )}
                <div className="clip-header">
                  <h3>{clip.title || `Clip ${index + 1}`}</h3>
                  {clip.score != null && (
                    <span className="badge">⚡ {clip.score}</span>
                  )}
                </div>
                <div className="clip-timestamp">
                  {formatTime(clip.start)} &ndash; {formatTime(clip.end)}
                </div>
                {clip.text && (
                  <p className="clip-transcript">"{clip.text}"</p>
                )}
                <a className="clip-download" href={clip.url} download>
                  ↓ Download Clip
                </a>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default App
