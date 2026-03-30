import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Subpath hosting: set VITE_BASE_PATH=/subdir/ when the app is not served from domain root
const base = process.env.VITE_BASE_PATH || '/'

// https://vite.dev/config/
export default defineConfig({
  base,
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
    },
  },
})
