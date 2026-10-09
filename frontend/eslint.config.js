import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores([
    'dist',
    // Legacy screens and compatibility transports are not part of the active App flow.
    'src/components/analysis/**',
    'src/components/chat/**',
    'src/components/results/**',
    'src/components/workspace/**',
    'src/hooks/**',
    'src/layouts/**',
    'src/pages/**',
    'src/services/websocket.ts',
    'src/store/**',
    'src/types/**',
  ]),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
  },
])
