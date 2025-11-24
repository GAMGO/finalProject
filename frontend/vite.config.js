import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 프론트엔드가 5173 포트에서 실행 중일 때,
    proxy: {
      // '/api'로 시작하는 모든 요청을 백엔드 서버로 포워딩
      '/api': {
        // 여기에 백엔드 서버의 실제 주소를 입력하세요 (예: http://localhost:8080)
        target: 'http://localhost:8080', 
        changeOrigin: true,
        secure: false, // HTTPS가 아닐 경우
        // rewrite: (path) => path.replace(/^\/api/, '') // 백엔드 경로에 /api가 포함되지 않는다면 필요
      },
    },
  },
})
