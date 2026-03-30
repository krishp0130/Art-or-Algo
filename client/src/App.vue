<script setup>
import { ref } from 'vue'
import AboutProject from './components/AboutProject.vue'
import Classifier from './components/Classifier.vue'
import FailureGallery from './components/FailureGallery.vue'
import MetricsDashboard from './components/MetricsDashboard.vue'

const tab = ref('app')

function goApp() {
  tab.value = 'app'
}

function goAbout() {
  tab.value = 'about'
}
</script>

<template>
  <div class="page">
    <nav class="tabs" aria-label="Primary">
      <button
        type="button"
        role="tab"
        :aria-selected="tab === 'app'"
        :class="{ active: tab === 'app' }"
        @click="goApp"
      >
        App
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="tab === 'about'"
        :class="{ active: tab === 'about' }"
        @click="goAbout"
      >
        About
      </button>
    </nav>

    <template v-if="tab === 'app'">
      <header class="hero">
        <p class="eyebrow">05-318 · Art or Algorithm</p>
        <h1>Human eyes <span class="amp">&amp;</span> machine judgment</h1>
        <p class="intro">
          Upload art, record your guess before any model output, then reveal the ViT’s call,
          confidence, and attention overlay—plus training metrics and known failure modes.
        </p>
      </header>

      <main class="main">
        <Classifier />
        <FailureGallery />
        <MetricsDashboard />
      </main>

      <footer class="foot">
        <p>
          Production: <code>npm start</code> from the repo root builds the Vue app and serves
          <code>client/dist</code> with Express. Dev: <code>npm run dev</code> in
          <code>client/</code> with the API on port 3000 (Vite proxies <code>/api</code>).
        </p>
      </footer>
    </template>

    <template v-else>
      <header class="hero compact">
        <p class="eyebrow">05-318 · Art or Algorithm</p>
        <h1>Documentation</h1>
        <p class="intro">
          How we built the dataset pipeline, ViT model, inference bridge, and Vue interface.
        </p>
      </header>
      <main class="main about-wrap">
        <AboutProject />
      </main>
      <footer class="foot">
        <p>
          For the latest commands and paths, see the repository <code>README.md</code>.
        </p>
      </footer>
    </template>
  </div>
</template>

<style scoped>
.page {
  max-width: 960px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
}

.tabs {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.75rem;
  padding: 0.35rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  width: fit-content;
  box-shadow: 0 4px 18px rgba(26, 22, 20, 0.06);
}

.tabs button {
  appearance: none;
  border: none;
  background: transparent;
  color: var(--muted);
  font-family: var(--font-body);
  font-size: 0.95rem;
  font-weight: 700;
  padding: 0.5rem 1.25rem;
  border-radius: 999px;
  cursor: pointer;
  transition:
    background 0.15s,
    color 0.15s;
}

.tabs button:hover {
  color: var(--ink);
}

.tabs button.active {
  background: var(--accent);
  color: var(--surface);
}

.tabs button:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.hero {
  margin-bottom: 2.5rem;
}

.hero.compact {
  margin-bottom: 1.75rem;
}

.eyebrow {
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.5rem;
}

.hero h1 {
  margin: 0 0 0.75rem;
  font-size: clamp(1.85rem, 4vw, 2.55rem);
}

.amp {
  font-style: italic;
  color: var(--accent);
}

.intro {
  margin: 0;
  max-width: 56ch;
  color: var(--muted);
  font-size: 1.08rem;
}

.main {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.about-wrap {
  margin-bottom: 2rem;
}

.foot {
  margin-top: 2rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.88rem;
  color: var(--muted);
}

.foot code {
  font-size: 0.85em;
  background: var(--accent-soft);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}
</style>
