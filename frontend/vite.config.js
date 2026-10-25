import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
const api = ['/health', '/studies', '/anomaly-detection', '/vqa', '/temporal-vqa', '/results', '/demo']
export default defineConfig({
  plugins: [react()],
  server: { proxy: Object.fromEntries(api.map(p => [p, 'http://localhost:8000'])) },
})
