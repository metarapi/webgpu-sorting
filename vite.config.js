import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import glsl from 'vite-plugin-glsl';
import { resolve } from 'path';

export default defineConfig({
  base: '/webgpu-sorting/',
  plugins: [
    tailwindcss(),
    glsl({
      include: ['**/*.wgsl', '**/*.glsl', '**/*.vert', '**/*.frag'],
      defaultExtension: 'wgsl',
      compress: false
    }),
  ],
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
      },
      output: {
        assetFileNames: (assetInfo) => {
          const name = assetInfo.names?.[0] || 'unknown';
          const info = name.split('.');
          const extType = info[info.length - 1];
          if (/wgsl/.test(extType)) {
            return 'src/shaders/[name][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        }
      }
    }
  },
  assetsInclude: ['**/*.wgsl'],
  server: {
    port: 3000,
  }
});
