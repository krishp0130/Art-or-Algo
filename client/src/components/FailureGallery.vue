<script setup>
import { computed, onMounted, ref } from 'vue'

const apiBase = import.meta.env.VITE_API_BASE ?? ''
const staticBase = import.meta.env.BASE_URL

const loading = ref(true)
const error = ref(null)
const failures = ref([])

function displayLabel(v) {
  const s = String(v || '').toLowerCase()
  return s === 'ai' ? 'AI' : 'Human'
}

function normalizeFailureRows(raw) {
  const arr = Array.isArray(raw?.failures) ? raw.failures : []
  return arr.map((f) => ({
    file: f.filename ?? f.file,
    trueLabel: f.true_label ?? f.trueLabel,
    predictedLabel: f.predicted_label ?? f.predictedLabel,
    confidence:
      f.confidence_in_predicted_class ?? f.confidence ?? null,
    hints: f.why_the_model_may_have_struggled ?? f.hints ?? [],
  }))
}

function imageUrl(file) {
  if (!file) return ''
  const enc = encodeURIComponent(file)
  if (apiBase) return `${apiBase}/api/failure-images/${enc}`
  return `${staticBase}failures/${enc}`
}

const hasItems = computed(() => failures.value.length > 0)

async function load() {
  loading.value = true
  error.value = null
  try {
    let data
    if (apiBase) {
      const res = await fetch(`${apiBase}/api/failures`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      data = await res.json()
    } else {
      const res = await fetch(`${staticBase}failures/failures.json`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      data = await res.json()
    }
    failures.value = normalizeFailureRows(data)
  } catch (e) {
    error.value = e.message || 'Could not load failures'
    failures.value = []
  } finally {
    loading.value = false
  }
}

onMounted(load)

function onImgError(e) {
  e.target.style.opacity = '0.35'
}
</script>

<template>
  <section class="gallery" aria-labelledby="fail-heading">
    <div class="head">
      <h2 id="fail-heading">Failure gallery</h2>
      <p class="sub">
        Examples where the model’s prediction disagreed with the human/AI label—useful for
        understanding what it gets wrong.
      </p>
      <button type="button" class="linkish" :disabled="loading" @click="load">
        {{ loading ? 'Loading…' : 'Refresh' }}
      </button>
    </div>

    <p v-if="loading" class="state">Loading…</p>
    <p v-else-if="error" class="state err">{{ error }}</p>
    <p v-else-if="!hasItems" class="state">
      No failure cases yet. Run <code>python3 ml/analyze_failures.py</code>, then
      <code>node scripts/sync-static-assets.mjs</code> before a static build—or use the
      Express API at <code>/api/failures</code>.
    </p>

    <ul v-else class="grid">
      <li v-for="(item, i) in failures" :key="item.file || i" class="card">
        <div class="thumb">
          <img
            :src="imageUrl(item.file)"
            :alt="`Misclassified sample ${item.file || i}`"
            loading="lazy"
            @error="onImgError"
          />
        </div>
        <dl class="meta">
          <div>
            <dt>True label</dt>
            <dd>{{ displayLabel(item.trueLabel) }}</dd>
          </div>
          <div>
            <dt>Model said</dt>
            <dd>{{ displayLabel(item.predictedLabel) }}</dd>
          </div>
          <div v-if="item.confidence != null" class="full">
            <dt>Confidence</dt>
            <dd>{{ (Number(item.confidence) * 100).toFixed(0) }}%</dd>
          </div>
          <div v-if="item.hints?.length" class="hints full">
            <dt>Why it may have failed</dt>
            <dd>
              <ul>
                <li v-for="(h, hi) in item.hints" :key="hi">{{ h }}</li>
              </ul>
            </dd>
          </div>
        </dl>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.gallery {
  margin-top: 3rem;
  padding-top: 2.5rem;
  border-top: 1px solid var(--border);
}

.head {
  margin-bottom: 1.25rem;
}

.head h2 {
  margin: 0 0 0.35rem;
  font-size: 1.5rem;
}

.sub {
  margin: 0 0 0.75rem;
  max-width: 62ch;
  color: var(--muted);
  font-size: 0.98rem;
}

.sub code {
  font-size: 0.85em;
  background: var(--accent-soft);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.linkish {
  appearance: none;
  border: none;
  background: none;
  color: var(--accent);
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.linkish:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.state {
  color: var(--muted);
}

.state.err {
  color: #8b2942;
}

.grid {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1.25rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(26, 22, 20, 0.06);
}

.thumb {
  aspect-ratio: 4 / 3;
  background: #e8e4dc;
  border-bottom: 1px solid var(--border);
}

.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.meta {
  margin: 0;
  padding: 1rem 1.1rem;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem 1rem;
}

.meta div.full {
  grid-column: 1 / -1;
}

.meta dt {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin: 0 0 0.2rem;
}

.meta dd {
  margin: 0;
  font-weight: 700;
  font-size: 1.05rem;
}

.hints dd {
  font-weight: 400;
  font-size: 0.82rem;
  line-height: 1.45;
  color: var(--muted);
}

.hints ul {
  margin: 0;
  padding-left: 1.1rem;
}
</style>
