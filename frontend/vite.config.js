import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/models': 'http://backend:5000',
      '/applications': 'http://backend:5000',
      '/select-model': 'http://backend:5000',
      '/select-application': 'http://backend:5000',
      '/create': 'http://backend:5000',
      '/execute': 'http://backend:5000',
      '/datasets': 'http://backend:5000',
      '/select-dataset': 'http://backend:5000',
      '/upload-dataset': 'http://backend:5000',
      '/workflows': 'http://backend:5000',
      '/results': 'http://backend:5000'
    }
  }
})
