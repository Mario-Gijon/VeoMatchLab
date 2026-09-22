const app = document.querySelector('#app')
const matchId = document.body.dataset.matchId
const clipStoragePrefix = 'detection-lab:clip:'

function element(name, text, className) {
  const node = document.createElement(name)
  if (text !== undefined) node.textContent = text
  if (className) node.className = className
  return node
}

function button(label, onClick, className = '') {
  const node = element('button', label, className)
  node.type = 'button'
  node.addEventListener('click', onClick)
  return node
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(seconds || 0))
  const minutes = Math.floor(total / 60)
  return `${String(minutes).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}

function parseTime(value) {
  const text = String(value).trim()
  if (/^\d+(?::\d{1,2})?$/.test(text)) {
    const [minutes, seconds] = text.split(':').map(Number)
    return seconds === undefined ? minutes : minutes * 60 + seconds
  }
  return Number.NaN
}

function loadClip(id) {
  try {
    const clip = JSON.parse(localStorage.getItem(`${clipStoragePrefix}${id}`) || 'null')
    return Number.isFinite(clip?.start_s) && Number.isFinite(clip?.duration_s) && clip.start_s >= 0 && clip.duration_s > 0 ? clip : null
  } catch {
    return null
  }
}

function saveClip(id, clip) {
  try {
    localStorage.setItem(`${clipStoragePrefix}${id}`, JSON.stringify({ match_id: id, start_s: clip.start_s, duration_s: clip.duration_s }))
    return true
  } catch {
    return false
  }
}

window.DetectionLabClipStore = { loadClip, saveClip }

function showError(message) {
  app.replaceChildren(element('p', message, 'message error'))
}

async function request(path) {
  const response = await fetch(path)
  if (!response.ok) throw new Error('No se pudieron cargar los datos.')
  return response.json()
}

async function requestJson(path, method, body) {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail || 'No se pudo iniciar el análisis.')
  }
  return response.json()
}

function appHeader(title, description) {
  const header = element('header', undefined, 'page-header')
  header.append(element('p', 'VeoMatchLab · desarrollo', 'eyebrow'), element('h1', title), element('p', description, 'subtitle'))
  return header
}

function resourcePanel(assets) {
  const panel = element('section', undefined, 'resource-panel')
  panel.append(element('h2', 'Recursos disponibles'))
  const grid = element('div', undefined, 'resource-grid')
  for (const asset of assets) {
    const card = element('div', undefined, `resource-card ${asset.available ? 'available' : 'missing'}`)
    card.append(element('span', asset.available ? 'Disponible' : 'No disponible', 'status'), element('strong', asset.label))
    grid.append(card)
  }
  panel.append(grid)
  return panel
}

async function showMatchList() {
  app.replaceChildren(appHeader('Detection Lab', 'Revisa partidos importados y prepara clips de trabajo para análisis visual.'))
  const matches = await request('/api/matches')
  const section = element('section', undefined, 'match-section')
  section.append(element('h2', 'Partidos importados'))
  if (!matches.length) {
    section.append(element('p', 'No hay partidos importados en el almacenamiento configurado.', 'message'))
  } else {
    const list = element('div', undefined, 'match-grid')
    for (const match of matches) {
      const card = element('article', undefined, 'match-card')
      const link = element('a', 'Abrir partido', 'primary-action')
      link.href = `/partidos/${encodeURIComponent(match.id)}`
      card.append(element('h3', match.title), element('p', match.id, 'match-id'), link)
      list.append(card)
    }
    section.append(list)
  }
  app.append(section)
}

function clipSummary(clip) {
  if (!clip) return element('p', 'Aún no has elegido un clip de trabajo.', 'message')
  const end = clip.start_s + clip.duration_s
  return element('p', `Clip actual: ${formatTime(clip.start_s)} → ${formatTime(end)} · duración ${formatTime(clip.duration_s)}`, 'clip-summary')
}

function previewVideo(match, clip) {
  if (!match.video_url) return element('p', 'El vídeo de doble lente no está disponible.', 'message error')
  const video = element('video', undefined, 'video-player')
  video.controls = true
  video.preload = 'metadata'
  video.src = match.video_url
  video.textContent = 'Tu navegador no puede reproducir este vídeo.'
  if (clip) video.addEventListener('loadedmetadata', () => { video.currentTime = clip.start_s }, { once: true })
  return video
}

function clipSelection(match, initialClip, onSaved) {
  const section = element('section', undefined, 'clip-panel')
  section.append(element('h2', 'Clip de trabajo'), element('p', 'Define un intervalo lógico para el futuro análisis; el vídeo original no se recorta ni se copia.', 'help-text'))
  const controls = element('div', undefined, 'clip-controls')
  const startLabel = element('label', 'Inicio (MM:SS)')
  const startInput = element('input')
  startInput.type = 'text'
  startInput.inputMode = 'numeric'
  startInput.value = formatTime(initialClip?.start_s || 0)
  startLabel.append(startInput)
  const durationLabel = element('label', 'Duración (MM:SS)')
  const durationInput = element('input')
  durationInput.type = 'text'
  durationInput.inputMode = 'numeric'
  durationInput.value = formatTime(initialClip?.duration_s || 30)
  durationLabel.append(durationInput)
  const endDisplay = element('p', '', 'derived-time')
  const feedback = element('p', '', 'form-feedback')
  function updateEnd() {
    const start = parseTime(startInput.value)
    const duration = parseTime(durationInput.value)
    endDisplay.textContent = Number.isFinite(start) && Number.isFinite(duration) && duration > 0 ? `Final: ${formatTime(start + duration)}` : 'Introduce tiempos válidos.'
  }
  startInput.addEventListener('input', updateEnd)
  durationInput.addEventListener('input', updateEnd)
  updateEnd()
  controls.append(startLabel, durationLabel, endDisplay)
  const save = button('Usar este clip', () => {
    const start_s = parseTime(startInput.value)
    const duration_s = parseTime(durationInput.value)
    if (!Number.isFinite(start_s) || start_s < 0 || !Number.isFinite(duration_s) || duration_s <= 0) {
      feedback.textContent = 'Usa valores MM:SS válidos y una duración mayor que cero.'
      feedback.className = 'form-feedback error'
      return
    }
    const clip = { match_id: match.id, start_s, duration_s }
    if (!saveClip(match.id, clip)) {
      feedback.textContent = 'No se pudo guardar el clip en este navegador.'
      feedback.className = 'form-feedback error'
      return
    }
    feedback.textContent = 'Clip de trabajo guardado en este navegador.'
    feedback.className = 'form-feedback success'
    onSaved(clip)
  }, 'primary-action')
  section.append(controls, save, feedback)
  return section
}

function visibleDetections(sample, threshold) {
  return (sample?.detections || []).filter((detection) => detection.confidence >= threshold)
}

function stackedBox(detection) {
  const [x1, y1, x2, y2] = detection.bbox
  const offset = detection.lens === 'right' ? 1024 : 0
  return [x1, y1 + offset, x2, y2 + offset]
}

function personPageState(clip, analysis) {
  if (!clip) return 'no-clip'
  return analysis?.state || 'not-run'
}

window.DetectionLabPersonView = { visibleDetections, stackedBox, personPageState }

function clipEnd(clip) {
  return clip.start_s + clip.duration_s
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value))
}

function relativeToAbsoluteTime(clip, relativeTime) {
  return clip.start_s + clamp(Number(relativeTime) || 0, 0, clip.duration_s)
}

function absoluteToRelativeTime(clip, absoluteTime) {
  return clamp((Number(absoluteTime) || 0) - clip.start_s, 0, clip.duration_s)
}

function clampToClip(clip, absoluteTime) {
  return clamp(Number(absoluteTime) || 0, clip.start_s, clipEnd(clip))
}

function clipPlaybackState(clip, absoluteTime) {
  const clampedTime = clampToClip(clip, absoluteTime)
  return {
    absoluteTime: clampedTime,
    relativeTime: absoluteToRelativeTime(clip, clampedTime),
    shouldPause: Number(absoluteTime) >= clipEnd(clip),
  }
}

function closestSample(samples, timestamp, tolerance) {
  let nearest = null
  let difference = Infinity
  for (const sample of samples || []) {
    const current = Math.abs(sample.timestamp_s - timestamp)
    if (current < difference) { nearest = sample; difference = current }
  }
  return difference <= tolerance ? nearest : null
}

function sampleAtClipTime(samples, clip, relativeTime, tolerance) {
  return closestSample(samples, relativeToAbsoluteTime(clip, relativeTime), tolerance)
}

window.DetectionLabClipPlayer = {
  relativeToAbsoluteTime,
  absoluteToRelativeTime,
  clampToClip,
  clipPlaybackState,
  sampleAtClipTime,
}

function clipVideo(match, clip) {
  if (!match.video_url) {
    const unavailable = element('p', 'El vídeo de doble lente no está disponible.', 'message error')
    return { video: unavailable, wrap: unavailable }
  }
  const video = element('video', undefined, 'video-player')
  video.controls = false
  video.preload = 'metadata'
  video.playsInline = true
  video.src = match.video_url
  video.textContent = 'Tu navegador no puede reproducir este vídeo.'
  const wrap = element('div', undefined, 'analysis-video-wrap')
  const stage = element('div', undefined, 'analysis-video-stage')
  const canvas = element('canvas', undefined, 'detection-overlay')
  canvas.setAttribute('aria-hidden', 'true')
  stage.append(video, canvas)
  const controls = element('div', undefined, 'clip-player-controls')
  const play = button('Reproducir', () => {
    if (video.currentTime >= clipEnd(clip)) video.currentTime = clip.start_s
    if (video.paused) video.play(); else video.pause()
  }, 'clip-play-button')
  const timeline = element('span', `00:00 / ${formatTime(clip.duration_s)}`, 'clip-player-time')
  const matchTimestamp = element('span', `Partido: ${formatTime(clip.start_s)}`, 'match-timestamp')
  const seek = element('input', undefined, 'clip-seek')
  seek.type = 'range'; seek.min = '0'; seek.max = String(clip.duration_s); seek.step = '0.01'; seek.value = '0'
  seek.setAttribute('aria-label', 'Posición dentro del clip')
  const volume = element('input', undefined, 'clip-volume')
  volume.type = 'range'; volume.min = '0'; volume.max = '1'; volume.step = '0.05'; volume.value = String(video.volume)
  volume.setAttribute('aria-label', 'Volumen')
  function updateControls(absoluteTime = video.currentTime) {
    const state = clipPlaybackState(clip, absoluteTime)
    timeline.textContent = `${formatTime(state.relativeTime)} / ${formatTime(clip.duration_s)}`
    matchTimestamp.textContent = `Partido: ${formatTime(state.absoluteTime)}`
    seek.value = String(state.relativeTime)
    play.textContent = video.paused ? 'Reproducir' : 'Pausa'
    return state
  }
  function enforceBounds() {
    const state = updateControls()
    if (video.currentTime !== state.absoluteTime) video.currentTime = state.absoluteTime
    if (state.shouldPause) video.pause()
  }
  video.addEventListener('loadedmetadata', () => { video.currentTime = clip.start_s; updateControls(clip.start_s) }, { once: true })
  video.addEventListener('timeupdate', enforceBounds)
  video.addEventListener('seeking', enforceBounds)
  video.addEventListener('play', () => { if (video.currentTime >= clipEnd(clip)) video.currentTime = clip.start_s; updateControls() })
  video.addEventListener('play', () => updateControls())
  video.addEventListener('pause', () => updateControls())
  seek.addEventListener('input', () => { video.currentTime = relativeToAbsoluteTime(clip, Number(seek.value)); updateControls(video.currentTime) })
  volume.addEventListener('input', () => { video.volume = Number(volume.value) })
  controls.append(play, timeline, seek, matchTimestamp, volume)
  wrap.append(stage, controls)
  return { video, canvas, wrap, controls }
}

function personTab(match, clip) {
  const panel = element('section', undefined, 'person-workspace')
  panel.append(element('h2', 'Person'), element('p', 'Detección espacial por lente sobre el clip lógico seleccionado. No incluye tracking ni proyección.', 'help-text'))
  panel.append(clipSummary(clip))
  if (personPageState(clip) === 'no-clip') {
    panel.append(element('p', 'Primero selecciona y guarda un clip en la pestaña Vista previa.', 'message'))
    return panel
  }
  const controls = element('div', undefined, 'analysis-controls')
  const modelLabel = element('label', 'Modelo')
  const model = element('select')
  const option = element('option', 'RF-DETR Large')
  option.value = 'rfdetr_large'
  model.append(option)
  modelLabel.append(model)
  const sampleLabel = element('label', 'Muestreo (FPS)')
  const sampleFps = element('input')
  sampleFps.type = 'number'; sampleFps.min = '0.1'; sampleFps.max = '30'; sampleFps.step = '0.5'; sampleFps.value = '5'
  sampleLabel.append(sampleFps)
  const batchLabel = element('label', 'Tamaño de batch')
  const batchSize = element('select')
  for (const size of [1, 2, 4, 8]) {
    const batchOption = element('option', String(size))
    batchOption.value = String(size)
    batchOption.selected = size === 8
    batchSize.append(batchOption)
  }
  batchLabel.append(batchSize)
  const inferenceLabel = element('label', 'Umbral de inferencia')
  const inference = element('input')
  inference.type = 'number'; inference.min = '0'; inference.max = '1'; inference.step = '0.05'; inference.value = '0.10'
  inferenceLabel.append(inference)
  const visibleLabel = element('label', 'Confianza visible: 0.25')
  const visible = element('input')
  visible.type = 'range'; visible.min = '0.10'; visible.max = '1'; visible.step = '0.05'; visible.value = '0.25'
  visibleLabel.append(visible)
  controls.append(modelLabel, sampleLabel, batchLabel, inferenceLabel, visibleLabel)
  const status = element('p', 'Análisis no ejecutado para este clip.', 'analysis-status')
  const debug = element('p', 'Reproduce el clip para ver las detecciones analizadas más cercanas.', 'frame-debug')
  const media = clipVideo(match, clip)
  let result = null
  let polling = null

  function draw() {
    if (!media.canvas || !result || !media.video.videoWidth) return
    const canvas = media.canvas
    canvas.width = media.video.videoWidth; canvas.height = media.video.videoHeight
    const context = canvas.getContext('2d')
    context.clearRect(0, 0, canvas.width, canvas.height)
    const analysis = result.analysis || {}
    // video.currentTime remains the real match timestamp; only the player UI is relative.
    const matchTime = clampToClip(clip, media.video.currentTime)
    const sample = closestSample(result.samples, matchTime, Math.max(0.2, 0.6 / (analysis.sample_fps || 5)))
    const detections = visibleDetections(sample, Number(visible.value))
    let left = 0; let right = 0
    context.lineWidth = 3; context.font = '20px system-ui'; context.textBaseline = 'top'
    for (const detection of detections) {
      const [x1, y1, x2, y2] = stackedBox(detection)
      if (detection.lens === 'left') left += 1; else right += 1
      context.strokeStyle = '#56e08a'; context.fillStyle = '#56e08a'
      context.strokeRect(x1, y1, x2 - x1, y2 - y1)
      context.fillText(detection.confidence.toFixed(2), x1 + 3, Math.max(0, y1 - 22))
    }
    const timestamp = sample ? formatTime(sample.timestamp_s) : 'sin muestra cercana'
    debug.textContent = `Muestra: ${timestamp} · izquierda: ${left} · derecha: ${right} · total visible: ${detections.length} · ${analysis.sample_fps || '?'} FPS`
  }
  visible.addEventListener('input', () => { visibleLabel.firstChild.textContent = `Confianza visible: ${Number(visible.value).toFixed(2)}`; draw() })
  if (media.video.tagName === 'VIDEO') {
    media.video.addEventListener('timeupdate', draw)
    media.video.addEventListener('seeked', draw)
    media.video.addEventListener('loadedmetadata', draw)
  }
  const analyze = button('Analizar clip', async () => {
    analyze.disabled = true; status.textContent = 'Procesando análisis de personas…'; status.className = 'analysis-status processing'
    try {
      const response = await requestJson(`/api/matches/${encodeURIComponent(match.id)}/person/analyses`, 'POST', {
        model_id: model.value, start_s: clip.start_s, duration_s: clip.duration_s,
        sample_fps: Number(sampleFps.value), batch_size: Number(batchSize.value),
        inference_threshold: Number(inference.value),
      })
      if (response.state === 'cached') {
        result = response.result; status.textContent = 'Resultado en caché reutilizado.'; status.className = 'analysis-status cached'; draw(); return
      }
      const jobId = response.job_id
      polling = window.setInterval(async () => {
        try {
          const current = await request(`/api/person/analyses/${encodeURIComponent(jobId)}`)
          if (current.state === 'processing') {
            const progress = current.progress
            if (progress) status.textContent = `Procesando ${progress.processed_frames} / ${progress.total_frames} frames (${progress.percent}%)…`
            return
          }
          window.clearInterval(polling); polling = null
          if (current.state === 'error') throw new Error(current.error || 'El análisis falló.')
          result = current.result; status.textContent = 'Análisis completo.'; status.className = 'analysis-status complete'; analyze.disabled = false; draw()
        } catch (error) {
          if (polling) window.clearInterval(polling)
          polling = null; status.textContent = error.message; status.className = 'analysis-status error'; analyze.disabled = false
        }
      }, 800)
    } catch (error) {
      status.textContent = error.message; status.className = 'analysis-status error'
    } finally {
      if (!polling) analyze.disabled = false
    }
  }, 'primary-action')
  panel.append(controls, analyze, status, media.wrap, debug)
  return panel
}

function tabButton(label, active, onClick) {
  const node = button(label, onClick, 'tab')
  node.setAttribute('role', 'tab')
  node.setAttribute('aria-selected', String(active))
  if (active) node.classList.add('active')
  return node
}

async function showMatchWorkspace(id) {
  const match = await request(`/api/matches/${encodeURIComponent(id)}`)
  let clip = loadClip(match.id)
  let selectedTab = new URLSearchParams(window.location.search).get('tab') === 'person' ? 'person' : 'preview'
  app.replaceChildren()
  const back = element('a', '← Volver a partidos', 'back-link')
  back.href = '/'
  const header = appHeader(match.title, `ID: ${match.id}`)
  const tabs = element('nav', undefined, 'tabs')
  tabs.setAttribute('aria-label', 'Secciones del partido')
  const content = element('div', undefined, 'tab-content')

  function render() {
    tabs.replaceChildren(
      tabButton('Vista previa', selectedTab === 'preview', () => { selectedTab = 'preview'; history.replaceState(null, '', `/partidos/${encodeURIComponent(match.id)}`); render() }),
      tabButton('Person', selectedTab === 'person', () => { selectedTab = 'person'; history.replaceState(null, '', `/partidos/${encodeURIComponent(match.id)}?tab=person`); render() }),
    )
    content.replaceChildren()
    if (selectedTab === 'preview') {
      content.append(previewVideo(match, clip), clipSelection(match, clip, (savedClip) => { clip = savedClip; render() }), clipSummary(clip))
    } else {
      content.append(personTab(match, clip))
    }
  }

  app.append(back, header, resourcePanel(match.assets), tabs, content)
  render()
}

(async () => {
  try {
    if (matchId) await showMatchWorkspace(matchId)
    else await showMatchList()
  } catch (error) {
    showError(error instanceof Error ? error.message : 'No se pudieron cargar los datos.')
  }
})()
