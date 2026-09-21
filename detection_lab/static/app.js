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

function personTab(match, clip) {
  const panel = element('section', undefined, 'person-workspace')
  panel.append(element('h2', 'Person'), element('p', 'Espacio de preparación para el futuro análisis de personas.', 'help-text'))
  panel.append(clipSummary(clip))
  if (!clip) {
    panel.append(element('p', 'Primero selecciona y guarda un clip en la pestaña Vista previa.', 'message'))
    return panel
  }
  panel.append(previewVideo(match, clip))
  const future = element('div', undefined, 'future-panel')
  future.append(element('h3', 'Próximos controles'), element('p', 'Aquí se añadirán modelo, umbral y ejecución de análisis. Todavía no se carga ningún detector.'))
  const controls = element('div', undefined, 'future-controls')
  const model = element('button', 'Modelo (próximamente)')
  const threshold = element('button', 'Umbral (próximamente)')
  model.disabled = true
  threshold.disabled = true
  controls.append(model, threshold)
  future.append(controls)
  panel.append(future)
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
