<script setup>
import { computed, onMounted, ref } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { Line } from 'vue-chartjs'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const apiBase = import.meta.env.VITE_API_BASE ?? ''
const staticBase = import.meta.env.BASE_URL

const loading = ref(true)
const error = ref(null)
const metrics = ref(null)

const chartFont = {
  family: "'Source Sans 3', system-ui, sans-serif",
  size: 12,
}

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: {
      position: 'bottom',
      labels: { font: chartFont, color: '#5c554e' },
    },
    tooltip: {
      titleFont: chartFont,
      bodyFont: chartFont,
    },
  },
  scales: {
    x: {
      grid: { color: 'rgba(26,22,20,0.06)' },
      ticks: { font: chartFont, color: '#5c554e' },
    },
    y: {
      grid: { color: 'rgba(26,22,20,0.06)' },
      ticks: { font: chartFont, color: '#5c554e' },
    },
  },
}

const lossChart = computed(() => {
  const m = metrics.value
  if (!m?.history?.length) return null
  const labels = m.history.map((h) => `Epoch ${h.epoch}`)
  return {
    labels,
    datasets: [
      {
        label: 'Train loss',
        data: m.history.map((h) => h.train_loss),
        borderColor: '#b54d30',
        backgroundColor: 'rgba(181, 77, 48, 0.12)',
        fill: true,
        tension: 0.25,
      },
      {
        label: 'Validation loss',
        data: m.history.map((h) => h.val_loss),
        borderColor: '#1a1614',
        backgroundColor: 'rgba(26, 22, 20, 0.06)',
        fill: true,
        tension: 0.25,
      },
    ],
  }
})

const lossOptions = {
  ...commonOptions,
  plugins: {
    ...commonOptions.plugins,
    title: {
      display: true,
      text: 'Loss per epoch',
      font: { ...chartFont, size: 14, weight: '600' },
      color: '#1a1614',
    },
  },
}

const accChart = computed(() => {
  const m = metrics.value
  if (!m?.history?.length) return null
  const labels = m.history.map((h) => `Epoch ${h.epoch}`)
  return {
    labels,
    datasets: [
      {
        label: 'Train accuracy',
        data: m.history.map((h) => h.train_acc * 100),
        borderColor: '#b54d30',
        backgroundColor: 'rgba(181, 77, 48, 0.08)',
        fill: true,
        tension: 0.25,
      },
      {
        label: 'Validation accuracy',
        data: m.history.map((h) => h.val_acc * 100),
        borderColor: '#2d6a4f',
        backgroundColor: 'rgba(45, 106, 79, 0.08)',
        fill: true,
        tension: 0.25,
      },
    ],
  }
})

const accOptions = {
  ...commonOptions,
  plugins: {
    ...commonOptions.plugins,
    title: {
      display: true,
      text: 'Accuracy per epoch (%)',
      font: { ...chartFont, size: 14, weight: '600' },
      color: '#1a1614',
    },
  },
  scales: {
    ...commonOptions.scales,
    y: {
      ...commonOptions.scales.y,
      min: 0,
      max: 100,
    },
  },
}

async function load() {
  loading.value = true
  error.value = null
  try {
    const remote = apiBase.trim()
    const primary = remote ? `${remote}/api/stats` : '/api/stats'
    let res = await fetch(primary)
    if (!res.ok) {
      res = await fetch(`${staticBase}metrics.json`)
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.error || `HTTP ${res.status}`)
    }
    metrics.value = await res.json()
  } catch (e) {
    error.value = e.message || 'Could not load metrics'
    metrics.value = null
  } finally {
    loading.value = false
  }
}

onMounted(load)

defineExpose({ reload: load })
</script>

<template>
  <section class="dashboard" aria-labelledby="dash-heading">
    <div class="head">
      <h2 id="dash-heading">Aggregate performance metrics</h2>
      <p class="sub">
        Training run summary from <code>metrics.json</code> (served by
        <code>GET /api/stats</code>).
      </p>
      <button type="button" class="linkish" :disabled="loading" @click="load">
        {{ loading ? 'Loading…' : 'Refresh' }}
      </button>
    </div>

    <p v-if="loading" class="state">Loading metrics…</p>
    <p v-else-if="error" class="state err">{{ error }}</p>

    <template v-else-if="metrics">
      <dl class="stats">
        <div>
          <dt>Best validation accuracy</dt>
          <dd>
            {{
              (
                (metrics.best_val_acc ?? metrics.final_accuracy ?? 0) * 100
              ).toFixed(2)
            }}%
          </dd>
        </div>
        <div>
          <dt>Epochs completed</dt>
          <dd>{{ metrics.epochs_ran }}</dd>
        </div>
        <div>
          <dt>Training time</dt>
          <dd>{{ metrics.train_seconds?.toFixed(0) ?? '—' }}s</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{{ metrics.model }}</dd>
        </div>
      </dl>

      <div v-if="lossChart && accChart" class="charts">
        <div class="chart-box">
          <Line :data="lossChart" :options="lossOptions" />
        </div>
        <div class="chart-box">
          <Line :data="accChart" :options="accOptions" />
        </div>
      </div>
      <p v-else class="state">No epoch history to chart.</p>
    </template>
  </section>
</template>

<style scoped>
.dashboard {
  margin-top: 3rem;
  padding-top: 2.5rem;
  border-top: 1px solid var(--border);
}

.head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem 1.25rem;
  margin-bottom: 1.25rem;
}

.head h2 {
  margin: 0;
  font-size: 1.5rem;
  width: 100%;
}

.sub {
  margin: 0;
  flex: 1;
  min-width: 200px;
  color: var(--muted);
  font-size: 0.95rem;
}

.sub code {
  font-size: 0.85em;
  background: var(--accent-soft);
  padding: 0.12em 0.35em;
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

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem 1.5rem;
  margin: 0 0 1.75rem;
}

.stats div {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.15rem;
  box-shadow: 0 4px 20px rgba(26, 22, 20, 0.05);
}

.stats dt {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin: 0 0 0.35rem;
}

.stats dd {
  margin: 0;
  font-family: var(--font-display);
  font-size: 1.35rem;
  font-weight: 700;
}

.charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
  gap: 1.5rem;
}

.chart-box {
  height: 280px;
  padding: 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
</style>
