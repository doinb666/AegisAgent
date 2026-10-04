import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

export default defineConfig({
  publicDir: false,
  build: {
    target: "es2022",
    outDir: fileURLToPath(new URL("../app/web", import.meta.url)),
    emptyOutDir: false,
    sourcemap: false,
    rollupOptions: {
      input: fileURLToPath(new URL("src/app.ts", import.meta.url)),
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]"
      }
    }
  }
});
