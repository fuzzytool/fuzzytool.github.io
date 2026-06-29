import { defineConfig } from "vite";

// Tauri expects a fixed port and serves the built assets from ../dist.
export default defineConfig({
  clearScreen: false,
  server: { port: 1420, strictPort: true },
  build: { outDir: "dist", target: "es2021" },
});
