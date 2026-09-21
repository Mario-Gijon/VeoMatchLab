const app = document.querySelector('#app')
const matchId = document.body.dataset.matchId

function element(name, text) {
  const node = document.createElement(name)
  if (text !== undefined) node.textContent = text
  return node
}

function heading(text) {
  const title = element('h1', text)
  app.append(title)
}

function showError(message) {
  app.replaceChildren(element('p', message))
}

async function request(path) {
  const response = await fetch(path)
  if (!response.ok) throw new Error('No se pudieron cargar los datos.')
  return response.json()
}

async function showMatchList() {
  heading('Detection Lab')
  app.append(element('p', 'Selecciona un partido importado para revisar sus recursos.'))
  const matches = await request('/api/matches')
  if (!matches.length) {
    app.append(element('p', 'No hay partidos importados en el almacenamiento configurado.'))
    return
  }
  const list = element('ul')
  for (const match of matches) {
    const item = element('li')
    const link = element('a', match.title)
    link.href = `/partidos/${encodeURIComponent(match.id)}`
    item.append(link, element('small', match.id))
    list.append(item)
  }
  app.append(list)
}

async function showPreview(id) {
  const match = await request(`/api/matches/${encodeURIComponent(id)}`)
  const back = element('a', '← Volver a partidos')
  back.href = '/'
  app.append(back)
  heading(match.title)
  app.append(element('p', `ID: ${match.id}`))

  if (match.video_url) {
    const video = element('video')
    video.controls = true
    video.preload = 'metadata'
    video.src = match.video_url
    video.textContent = 'Tu navegador no puede reproducir este vídeo.'
    app.append(video)
  } else {
    app.append(element('p', 'El vídeo de doble lente no está disponible.'))
  }

  app.append(element('h2', 'Recursos'))
  const assets = element('ul')
  for (const asset of match.assets) {
    assets.append(element('li', `${asset.label}: ${asset.available ? 'disponible' : 'no disponible'}`))
  }
  app.append(assets)
}

(async () => {
  try {
    if (matchId) await showPreview(matchId)
    else await showMatchList()
  } catch (error) {
    showError(error instanceof Error ? error.message : 'No se pudieron cargar los datos.')
  }
})()
