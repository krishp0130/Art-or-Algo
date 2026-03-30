<script setup>
const kaggleUrl =
  'https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art'
</script>

<template>
  <article class="about" aria-labelledby="about-title">
    <header class="about-head">
      <h2 id="about-title">About this project</h2>
      <p class="lede">
        CMU <strong>05-318</strong> — <em>Art or Algorithm</em>: a binary image classifier plus web
        app that separates human-made artwork from AI-generated art, with attention visualization
        and human-in-the-loop interaction design.
      </p>
    </header>

    <section class="block">
      <h3>1. Data pipeline</h3>
      <ol class="steps">
        <li>
          <strong>Source:</strong>
          <a :href="kaggleUrl" target="_blank" rel="noopener noreferrer"
            >Kaggle — AI art VS Human art</a
          >
          (CC0-1.0).
        </li>
        <li>
          <strong>Download:</strong> run <code>scripts/download_dataset.sh</code> (Kaggle API
          credentials required).
        </li>
        <li>
          <strong>Train / eval split:</strong>
          <code>python3 scripts/split_train_eval.py</code> produces stratified
          <code>data/train</code> and <code>data/eval</code> (see <code>data/README.md</code>).
        </li>
        <li>
          <strong>Loading in PyTorch:</strong> <code>ml/dataset.py</code> builds dataloaders with
          ViT-friendly preprocessing (resize, normalize to ImageNet stats).
        </li>
      </ol>
    </section>

    <section class="block">
      <h3>2. Model &amp; training</h3>
      <ol class="steps">
        <li>
          <strong>Architecture:</strong> Vision Transformer
          <strong>ViT-B/16</strong> from torchvision, pretrained on ImageNet; the classification
          head is replaced for <strong>two classes</strong> (<code>ai</code> vs
          <code>human</code>).
        </li>
        <li>
          <strong>Training script:</strong> <code>ml/train.py</code> (legacy
          <code>python3 -m ml.train</code>) or the master runner
          <code>python3 run_train.py</code> as described in the repo README.
        </li>
        <li>
          <strong>Optimization:</strong> AdamW, cosine LR schedule, cross-entropy loss; optional AMP
          where supported (CUDA/MPS).
        </li>
        <li>
          <strong>Outputs:</strong> checkpoint under <code>models/</code> (e.g.
          <code>final_vit.pth</code> or <code>best_vit.pth</code>) and
          <code>models/metrics.json</code> with validation accuracy, loss history, and related
          metadata for the dashboard charts.
        </li>
      </ol>
    </section>

    <section class="block">
      <h3>3. Interpretability (attention)</h3>
      <ol class="steps">
        <li>
          <strong>Attention rollout</strong> across ViT encoder blocks yields a patch importance
          vector; it is upsampled and blended over the <strong>original-resolution</strong> image
          (colormap + alpha) in <code>ml/inference.py</code>.
        </li>
        <li>
          The backend receives a <strong>base64 PNG</strong> of that overlay so the UI can show
          which regions the model emphasized—aligned with “transparency” goals for the course HCI
          narrative.
        </li>
      </ol>
    </section>

    <section class="block">
      <h3>4. Backend (Node.js)</h3>
      <ol class="steps">
        <li>
          <strong>Express</strong> in <code>server/</code> accepts multipart uploads (Multer),
          writes a temp file, and runs inference via
          <code>child_process.spawn</code>:
          <code>python -m ml.inference --image &lt;path&gt;</code> (use
          <code>PYTHON_BIN</code> pointing at your <code>.venv</code> when needed).
        </li>
        <li>
          <strong>JSON on stdout</strong> is parsed and returned as
          <code>POST /api/classify</code> (alias <code>/api/predict</code>): prediction, confidence,
          probabilities, and attention image data for the client.
        </li>
        <li>
          <strong>Other routes:</strong> <code>GET /api/stats</code> serves training metrics;
          <code>GET /api/failures</code> + static <code>/api/failure-images/</code> power the
          failure gallery from <code>models/failures.json</code> and
          <code>models/failure_gallery/</code>.
        </li>
        <li>
          <strong>Production:</strong> root <code>npm start</code> builds the Vue app and serves
          <code>client/dist</code> from the same server.
        </li>
      </ol>
    </section>

    <section class="block">
      <h3>5. Frontend (Vue 3)</h3>
      <ol class="steps">
        <li>
          <strong>Stack:</strong> Vite + Vue 3; Chart.js + vue-chartjs for aggregate metrics.
        </li>
        <li>
          <strong>Classifier flow:</strong> upload → user chooses Human vs AI → inference runs →
          optional <strong>reveal</strong> step before showing the model label and confidence as a
          percentage; original image and attention overlay are shown side by side with a short
          “why” tooltip.
        </li>
        <li>
          <strong>Failure gallery:</strong> surfaces misclassified examples (true label vs model
          prediction) when the ML pipeline exports them.
        </li>
        <li>
          <strong>Dev:</strong> <code>npm run dev</code> in <code>client/</code> with the API on
          port 3000 — Vite proxies <code>/api</code> to avoid CORS friction.
        </li>
      </ol>
    </section>

    <section class="block">
      <h3>6. Failure analysis (optional)</h3>
      <p class="para">
        High-confidence errors on the validation set can be exported for HCI critique (see
        <code>ml/analyze_failures.py</code> and the README). Those assets feed the failure gallery
        when paths are listed in <code>models/failures.json</code>.
      </p>
    </section>

    <footer class="about-foot">
      <p>
        Repository layout highlights: <code>ml/</code> training &amp; inference,
        <code>client/</code> Vue app, <code>server/</code> Express API,
        <code>scripts/</code> data prep, <code>models/</code> weights &amp; metrics.
      </p>
    </footer>
  </article>
</template>

<style scoped>
.about {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2rem 1.75rem 1.75rem;
  box-shadow: var(--shadow);
  max-width: 720px;
  margin: 0 auto;
}

.about-head {
  margin-bottom: 1.75rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid var(--border);
}

.about-head h2 {
  margin: 0 0 0.65rem;
  font-size: clamp(1.5rem, 3vw, 1.85rem);
}

.lede {
  margin: 0;
  font-size: 1.05rem;
  line-height: 1.6;
  color: var(--muted);
}

.block {
  margin-bottom: 1.65rem;
}

.block h3 {
  margin: 0 0 0.65rem;
  font-size: 1.12rem;
  font-family: var(--font-display);
  color: var(--ink);
}

.steps {
  margin: 0;
  padding-left: 1.35rem;
  color: var(--ink);
  line-height: 1.65;
}

.steps li {
  margin-bottom: 0.55rem;
}

.steps li::marker {
  color: var(--accent);
  font-weight: 700;
}

.steps code {
  font-size: 0.88em;
  background: var(--accent-soft);
  padding: 0.12em 0.4em;
  border-radius: 4px;
}

.steps a {
  font-weight: 600;
}

.para {
  margin: 0;
  line-height: 1.65;
  color: var(--muted);
}

.para code {
  font-size: 0.9em;
  background: var(--accent-soft);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.about-foot {
  margin-top: 1.5rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--border);
}

.about-foot p {
  margin: 0;
  font-size: 0.92rem;
  color: var(--muted);
  line-height: 1.55;
}

.about-foot code {
  font-size: 0.88em;
  background: var(--accent-soft);
  padding: 0.08em 0.35em;
  border-radius: 4px;
}
</style>
