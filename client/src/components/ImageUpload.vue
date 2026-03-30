<script setup>
import { ref } from 'vue'

const emit = defineEmits(['select'])

const props = defineProps({
  disabled: { type: Boolean, default: false },
})

const inputRef = ref(null)

function pick() {
  if (props.disabled) return
  inputRef.value?.click()
}

function onChange(e) {
  const f = e.target.files?.[0]
  if (f) emit('select', f)
  e.target.value = ''
}

function onDrop(e) {
  e.preventDefault()
  const f = e.dataTransfer?.files?.[0]
  if (!f || !/^image\//i.test(f.type)) return
  emit('select', f)
}

function onDragOver(e) {
  e.preventDefault()
}
</script>

<template>
  <div
    class="upload"
    :class="{ disabled }"
    @drop="onDrop"
    @dragover="onDragOver"
    role="button"
    tabindex="0"
    @keydown.enter.prevent="pick"
    @keydown.space.prevent="pick"
  >
    <input
      ref="inputRef"
      type="file"
      class="sr-only"
      accept="image/jpeg,image/png,image/webp,image/gif"
      :disabled="disabled"
      @change="onChange"
    />
    <div class="inner" @click="!disabled && pick()">
      <span class="glyph" aria-hidden="true">↗</span>
      <p class="lead"><strong>Drop an artwork</strong> or browse</p>
      <p class="hint">JPEG, PNG, WebP, or GIF · up to 25&nbsp;MB</p>
    </div>
  </div>
</template>

<style scoped>
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}

.upload {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow);
  transition:
    border-color 0.2s,
    background 0.2s;
}

.upload:not(.disabled):hover,
.upload:not(.disabled):focus-within {
  border-color: var(--accent);
  background: #fff;
}

.upload.disabled {
  opacity: 0.55;
  pointer-events: none;
}

.inner {
  padding: 2.25rem 1.5rem;
  text-align: center;
  cursor: pointer;
}

.glyph {
  display: block;
  font-size: 2rem;
  line-height: 1;
  margin-bottom: 0.75rem;
  opacity: 0.55;
  transform: rotate(-12deg);
}

.lead {
  margin: 0 0 0.35rem;
  color: var(--ink);
}

.hint {
  margin: 0;
  font-size: 0.9rem;
  color: var(--muted);
}
</style>
