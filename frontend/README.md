# MetaboAgent frontend

React, TypeScript, Vite, Zustand, AG Grid, and ECharts frontend for the
MetaboAgent API.

```bash
npm ci
cp .env.example .env
npm run dev
```

Available commands:

- `npm run dev` starts the development server.
- `npm run check` runs TypeScript and ESLint checks.
- `npm run build` creates a production build in `dist/`.
- `npm run preview` serves the production build locally.

Configure the API and WebSocket endpoints in `.env`.
