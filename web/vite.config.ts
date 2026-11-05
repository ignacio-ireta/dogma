/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The API runs on loopback (`dogma serve`, port 8765). In development the Vite server proxies
// /api there, so the browser sees one origin. A production build is served by the API itself.
const API_TARGET = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: false },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // The local API permits font-src 'self'; even small bundled fonts need file URLs.
    assetsInlineLimit: 0,
    sourcemap: false,
    target: "es2023",
  },
  test: {
    environment: "jsdom",
    globals: false,
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}", "scripts/**/*.test.mjs"],
    css: false,
  },
});
