import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    historyApiFallback: true,   // serve index.html for all unknown paths (SPA routing)
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/clips': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
