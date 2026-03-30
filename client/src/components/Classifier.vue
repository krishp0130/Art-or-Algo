<script setup>
import { computed, ref } from 'vue'
import ImageUpload from './ImageUpload.vue'

const apiBase = import.meta.env.VITE_API_BASE ?? ''
const classifyUrl = () => {
  const b = apiBase.trim()
  if (b) return `${b}/api/classify`
  return '/api/classify'
}
const showStaticApiHint = Boolean(
  import.meta.env.PROD && !apiBase.trim()
)

/** @type {import('vue').Ref<'idle' | 'guess' | 'loading' | 'pendingReveal' | 'revealed'>} */
const phase = ref('idle')

const selectedFile = ref(null)
const previewUrl = ref('')
const userGuess = ref(null)

const predictLoading = ref(false)
const predictError = ref(null)
const prediction = ref(null)

function normalizeLabel(v) {
  const s = String(v || '').toLowerCase()
  if (s === 'ai') return 'ai'
  if (s === 'human') return 'human'
  return s
}

function displayLabel(pred) {
  return normalizeLabel(pred) === 'ai' ? 'AI-generated' : 'Human-made'
}

function onFileSelected(file) {
  predictError.value = null
  prediction.value = null
  userGuess.value = null
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
  phase.value = 'guess'
}

async function submitGuess(guess) {
  if (!selectedFile.value) return
  userGuess.value = guess
  predictError.value = null
  prediction.value = null
  phase.value = 'loading'
  predictLoading.value = true

  const fd = new FormData()
  fd.append('image', selectedFile.value)

  try {
    const res = await fetch(classifyUrl(), {
      method: 'POST',
      body: fd,
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      throw new Error(data.detail || data.error || `Request failed (${res.status})`)
    }
    prediction.value = data
    phase.value = 'pendingReveal'
  } catch (e) {
    predictError.value = e.message || 'Classification failed'
    phase.value = 'guess'
  } finally {
    predictLoading.value = false
  }
}

function revealModelJudgment() {
  phase.value = 'revealed'
}

function resetFlow() {
  phase.value = 'idle'
  userGuess.value = null
  prediction.value = null
  predictError.value = null
  selectedFile.value = null
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
}

const overlaySrc = computed(() => {
  const p = prediction.value
  if (!p) return ''
  if (p.heatmap_data_url) return p.heatmap_data_url
  if (p.heatmap_png_base64) {
    return `data:image/png;base64,${p.heatmap_png_base64}`
  }
  return ''
})

const confidencePercent = computed(() => {
  const c = Number(prediction.value?.confidence)
  if (Number.isNaN(c)) return '—'
  return `${Math.min(100, Math.max(0, c * 100)).toFixed(0)}%`
})

const modelNorm = computed(() => normalizeLabel(prediction.value?.prediction))
const guessNorm = computed(() => normalizeLabel(userGuess.value))
const guessMatchesModel = computed(
  () => phase.value === 'revealed' && modelNorm.value === guessNorm.value
)
</script>

<template>
  <section class="classifier panel" aria-label="Classifier">
    <p v-if="showStaticApiHint" class="api-hint" role="note">
      Static hosting without a same-origin API: classification needs a live backend. Build with
      <code>VITE_API_BASE=https://your-backend.example</code> pointing to the Express server, or run
      <code>npm start</code> locally for same-origin <code>/api</code>.
    </p>
    <template v-if="phase === 'idle'">
      <ImageUpload @select="onFileSelected" />
    </template>

    <template v-else-if="phase === 'guess' || phase === 'loading'">
      <div class="preview-block">
        <img
          v-if="previewUrl"
          class="preview"
          :src="previewUrl"
          alt="Selected artwork preview"
        />
      </div>
      <h2 class="step-title">Your guess first</h2>
      <p class="step-desc">
        The model’s label and confidence stay hidden until you commit. Was this made by a
        <strong>human</strong> or by <strong>AI</strong>?
      </p>
      <div class="guess-actions" role="group" aria-label="Your guess">
        <button
          type="button"
          class="guess human"
          :disabled="phase === 'loading'"
          @click="submitGuess('human')"
        >
          Human-made
        </button>
        <button
          type="button"
          class="guess ai"
          :disabled="phase === 'loading'"
          @click="submitGuess('ai')"
        >
          AI-generated
        </button>
      </div>
      <p v-if="phase === 'loading'" class="loading-msg" aria-live="polite">
        Running ViT inference…
      </p>
      <p v-if="predictError" class="err-msg" role="alert">{{ predictError }}</p>
      <button type="button" class="text-btn" :disabled="phase === 'loading'" @click="resetFlow">
        Choose a different image
      </button>
    </template>

    <template v-else-if="phase === 'pendingReveal' && prediction && previewUrl">
      <div class="preview-block small">
        <img class="preview" :src="previewUrl" alt="Your artwork" />
      </div>
      <p class="locked-guess">
        Your guess: <strong>{{ displayLabel(userGuess) }}</strong>
      </p>
      <p class="step-desc subtle">
        Inference is done, but the model’s answer is still hidden—open it when you’re ready.
      </p>
      <button type="button" class="reveal-btn" @click="revealModelJudgment">
        Reveal model judgment
      </button>
      <button type="button" class="text-btn" @click="resetFlow">Start over</button>
    </template>

    <template v-else-if="phase === 'revealed' && prediction && previewUrl">
      <div
        class="result-banner"
        :class="{ match: guessMatchesModel, mismatch: !guessMatchesModel }"
      >
        <p class="banner-label">Transparency</p>
        <p class="banner-main">
          You said <strong>{{ displayLabel(userGuess) }}</strong>. The model predicts
          <strong>{{ displayLabel(prediction.prediction) }}</strong> with
          <strong class="conf-pct">{{ confidencePercent }}</strong> confidence.
        </p>
        <p v-if="guessMatchesModel" class="banner-hint align">You and the model agree.</p>
        <p v-else class="banner-hint disagree">You and the model disagree—worth a second look.</p>
      </div>

      <div class="viz-head">
        <h3 class="viz-title">Attention over your image</h3>
        <div class="why-wrap">
          <button
            type="button"
            class="why-btn"
            aria-describedby="cls-why-tip"
            aria-label="Why did it pick this?"
          >
            Why did it pick this?
          </button>
          <p id="cls-why-tip" class="why-tip" role="tooltip">
            The Vision Transformer splits the image into patches. Brighter areas in the overlay show
            where attention rolled out most strongly—those regions weighed more heavily in the
            model’s decision.
          </p>
        </div>
      </div>

      <div class="viz-grid">
        <figure class="viz-pane">
          <figcaption>Original</figcaption>
          <div class="viz-frame">
            <img :src="previewUrl" alt="Original upload" />
          </div>
        </figure>
        <figure class="viz-pane hero">
          <figcaption>Overlay (patches the ViT emphasized)</figcaption>
          <div class="viz-frame overlay-frame">
            <img v-if="overlaySrc" :src="overlaySrc" alt="Attention heatmap over the artwork" />
            <p v-else class="no-hm">No heatmap returned from the server.</p>
          </div>
        </figure>
      </div>

      <p v-if="overlaySrc" class="viz-note">
        The right panel blends the attention map onto your original pixels so you can see which
        parts of the composition the model treated as evidence.
      </p>

      <button type="button" class="primary-outline" @click="resetFlow">
        Try another artwork
      </button>
    </template>
  </section>
</template>

<style scoped>
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.75rem;
  box-shadow: var(--shadow);
}

.api-hint {
  margin: 0 0 1rem;
  padding: 0.75rem 1rem;
  font-size: 0.9rem;
  line-height: 1.45;
  color: var(--muted);
  background: var(--accent-soft);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}

.api-hint code {
  font-size: 0.85em;
  word-break: break-all;
}

.preview-block {
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  margin-bottom: 1.5rem;
  background: #e8e4dc;
  max-height: 320px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-block.small {
  max-height: 220px;
}

.preview {
  display: block;
  max-width: 100%;
  max-height: 320px;
  object-fit: contain;
}

.preview-block.small .preview {
  max-height: 220px;
}

.step-title {
  margin: 0 0 0.5rem;
  font-size: 1.35rem;
}

.step-desc {
  margin: 0 0 1.25rem;
  color: var(--muted);
}

.step-desc.subtle {
  font-size: 0.98rem;
}

.locked-guess {
  margin: 0 0 0.75rem;
  font-size: 1.1rem;
}

.guess-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1rem;
}

.guess {
  flex: 1;
  min-width: 140px;
  padding: 1rem 1.25rem;
  font-size: 1.05rem;
  font-weight: 700;
  border-radius: var(--radius);
  border: 2px solid var(--border);
  cursor: pointer;
  transition:
    transform 0.15s,
    box-shadow 0.15s,
    border-color 0.15s;
}

.guess:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.guess:not(:disabled):hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow);
}

.guess.human {
  background: #fff;
  color: var(--ink);
}

.guess.ai {
  background: var(--ink);
  color: var(--surface);
  border-color: var(--ink);
}

.reveal-btn {
  display: inline-block;
  margin-bottom: 0.75rem;
  padding: 0.85rem 1.5rem;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--surface);
  background: var(--accent);
  border: none;
  border-radius: var(--radius);
  cursor: pointer;
  box-shadow: var(--shadow);
  transition: filter 0.15s;
}

.reveal-btn:hover {
  filter: brightness(1.05);
}

.loading-msg {
  margin: 0 0 0.75rem;
  font-weight: 600;
  color: var(--accent);
}

.err-msg {
  margin: 0 0 0.75rem;
  color: #8b2942;
  font-weight: 600;
}

.text-btn {
  display: inline-block;
  margin-top: 0.25rem;
  padding: 0;
  border: none;
  background: none;
  color: var(--muted);
  font-size: 0.95rem;
  text-decoration: underline;
  cursor: pointer;
}

.text-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.result-banner {
  padding: 1.1rem 1.25rem;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  margin-bottom: 1.25rem;
}

.result-banner.match {
  background: rgba(45, 106, 79, 0.1);
  border-color: rgba(45, 106, 79, 0.35);
}

.result-banner.mismatch {
  background: rgba(181, 77, 48, 0.08);
  border-color: rgba(181, 77, 48, 0.35);
}

.banner-label {
  margin: 0 0 0.35rem;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

.banner-main {
  margin: 0;
  font-size: 1.08rem;
  line-height: 1.5;
}

.conf-pct {
  color: var(--accent);
  font-weight: 700;
}

.banner-hint {
  margin: 0.65rem 0 0;
  font-size: 0.95rem;
}

.banner-hint.align {
  color: #2d6a4f;
  font-weight: 600;
}

.banner-hint.disagree {
  color: var(--accent);
  font-weight: 600;
}

.viz-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem 1rem;
  margin-bottom: 1rem;
}

.viz-title {
  margin: 0;
  font-size: 1.2rem;
}

.why-wrap {
  position: relative;
}

.why-btn {
  appearance: none;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--accent);
  font-size: 0.88rem;
  font-weight: 600;
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  cursor: help;
  transition:
    background 0.15s,
    border-color 0.15s;
}

.why-btn:hover,
.why-btn:focus-visible {
  background: var(--accent-soft);
  border-color: var(--accent);
  outline: none;
}

.why-tip {
  display: none;
  position: absolute;
  z-index: 5;
  right: 0;
  top: calc(100% + 8px);
  width: min(340px, 88vw);
  margin: 0;
  padding: 0.85rem 1rem;
  font-size: 0.88rem;
  line-height: 1.45;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

.why-wrap:hover .why-tip,
.why-wrap:focus-within .why-tip {
  display: block;
}

.viz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1.25rem;
  margin-bottom: 0.75rem;
}

.viz-pane {
  margin: 0;
}

.viz-pane.hero {
  grid-column: span 1;
}

@media (min-width: 720px) {
  .viz-pane.hero {
    grid-column: span 1;
  }
}

.viz-pane figcaption {
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}

.viz-frame {
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  background: #e8e4dc;
  aspect-ratio: 4 / 3;
  display: flex;
  align-items: center;
  justify-content: center;
}

.overlay-frame {
  background: #1a1614;
}

.viz-frame img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  vertical-align: middle;
}

.no-hm {
  margin: 0;
  padding: 1rem;
  color: var(--muted);
  font-size: 0.9rem;
  text-align: center;
}

.viz-note {
  margin: 0 0 1.25rem;
  font-size: 0.92rem;
  color: var(--muted);
}

.primary-outline {
  margin-top: 0.25rem;
  padding: 0.65rem 1.25rem;
  font-weight: 700;
  font-size: 0.98rem;
  color: var(--accent);
  background: transparent;
  border: 2px solid var(--accent);
  border-radius: var(--radius);
  cursor: pointer;
  transition:
    background 0.15s,
    color 0.15s;
}

.primary-outline:hover {
  background: var(--accent);
  color: var(--surface);
}
</style>
