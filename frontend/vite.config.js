import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
const VITE_BASE_URL = process.env.VITE_BASE_URL;
const VITE_BASE_URL = process.env.VITE_BASE_URL;
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // 1. 🤖 AI 서버 API 프록시 (더 구체적인 경로를 먼저 정의)
      // '/api/data'로 시작하는 모든 요청을 로컬 AI 서버로 포워딩
      '/api/data': {
        target: `${VITE_BASE_URL}`, // ⭐️ 로컬 AI 서버 주소
        changeOrigin: true,
        secure: false, 
      },
      
      // 2. 🌸 Spring 백엔드 API 프록시
      // '/api'로 시작하는 나머지 모든 요청을 주 백엔드 서버로 포워딩
      '/api': {
        target: `${VITE_BASE_URL}`, 
        changeOrigin: true,
        secure: true,
      },
    },
  },
})
