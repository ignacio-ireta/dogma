const concepts = window.dogmaConcepts;
let active = new URLSearchParams(location.search).get('concept') || 'memphis';
if (!concepts.some(c => c.id === active)) active = 'memphis';
let screen = 'today';
let previewVersion = 0;
const nav = document.querySelector('#concepts');
nav.innerHTML = concepts.map(c => `<button class="concept-option" data-concept="${c.id}" data-choice="${c.id}" aria-pressed="false"><span class="option-number">${c.number}</span><span><span class="option-title">${c.title}</span><span class="option-ref">${c.id === 'memphis' ? 'BONUS · Corporate Memphis' : c.flavor}</span></span><span class="color-dot" style="background:${c.colors[2]}"></span></button>`).join('');
document.querySelector('#overview-grid').innerHTML = concepts.map(c => `<button class="overview-card" data-concept="${c.id}" data-choice="${c.id}"><img src="assets/previews/${c.id}.png" alt="${c.title} Today screen" loading="lazy"><span class="card-body"><h3>${c.number} / ${c.title}</h3><p>${c.tagline}</p><p class="scope">${c.scope}</p></span></button>`).join('');
document.querySelector('#source-list').innerHTML = window.dogmaSources.map(s => `<div class="source"><a href="${s.href}" target="_blank" rel="noopener noreferrer">${s.name} ↗</a><p>${s.text}</p></div>`).join('');
function update() {
  const c = concepts.find(c => c.id === active);
  document.querySelector('#concept-kicker').textContent = `${c.number} / ${c.id === 'memphis' ? 'BONUS · CORPORATE MEMPHIS' : c.reference.toUpperCase()}`;
  document.querySelector('#concept-title').textContent = c.title;
  document.querySelector('#concept-tagline').textContent = c.tagline;
  document.querySelector('#palette').innerHTML = c.colors.map(color => `<span style="background:${color}" title="${color}" aria-label="${color}"></span>`).join('');
  document.querySelector('#reasoning').innerHTML = [['WHAT CHANGES',c.changes],['WHY IT FITS',c.fit],['TRADE-OFF',c.tradeoff]].map(([h,p]) => `<div><strong>${h}</strong><p>${p}</p></div>`).join('');
  document.querySelectorAll('[data-concept]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.concept === active)));
  document.querySelectorAll('[data-screen]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.screen === screen)));
  const preview = document.querySelector('#preview');
  const query = `concept=${active}&screen=${screen}`;
  preview.dataset.query = query;
  const version = ++previewVersion;
  if (location.protocol === 'file:') {
    preview.src = `preview.html?${query}`;
  } else {
    // The review server intentionally forbids framing its responses. Render this local
    // preview document in srcdoc, with its assets resolved against the board's base URL.
    fetch('preview.html').then(response => {
      if (!response.ok) throw new Error('Could not load the preview');
      return response.text();
    }).then(html => {
      if (version !== previewVersion) return;
      preview.srcdoc = html.replace('<head>', `<head><base href="${new URL('.', document.baseURI).href}">`);
    }).catch(error => {
      if (version !== previewVersion) return;
      preview.srcdoc = `<p style="font-family:system-ui;padding:24px">${error.message}. Open the saved index.html file to review the concepts offline.</p>`;
    });
  }
  document.querySelector('#preview-address').textContent = `dogma / ${screen === 'today' ? 'today' : 'vector-projection'}`;
}
function setOverview(show) {
  document.querySelector('#overview').hidden = !show;
  document.querySelector('#detail').hidden = show;
  document.querySelector('#overview-toggle').setAttribute('aria-pressed',String(show));
}
document.addEventListener('click', e => {
  const concept = e.target.closest('[data-concept]');
  if (concept) { active = concept.dataset.concept; setOverview(false); update(); }
  const view = e.target.closest('[data-screen]');
  if (view) { screen = view.dataset.screen; update(); }
  const size = e.target.closest('[data-size]');
  if (size) {
    document.querySelector('#preview-window').classList.toggle('mobile',size.dataset.size === 'mobile');
    document.querySelectorAll('[data-size]').forEach(b => b.setAttribute('aria-pressed',String(b === size)));
  }
});
document.querySelector('#overview-toggle').addEventListener('click', () => setOverview(document.querySelector('#overview').hidden));
window.addEventListener('message', e => {
  if (e.source === document.querySelector('#preview').contentWindow && e.data?.type === 'dogma-preview-view' && ['today','workspace'].includes(e.data.screen)) { screen = e.data.screen; update(); }
});
update();
