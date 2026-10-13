import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js"

// EARS (the web app, :7860) serves this page's static build at /, the classic page at /classic, the WebSocket and the
// voice list. `npm run build` exports the static files to out/, which EARS serves (app/ears/web.py; run.sh builds it).
// `npm run dev` runs this page on :3000 for work on it, and passes the rest through to EARS (EARS_URL), so the page
// talks to one origin either way.
const EARS = process.env.EARS_URL || "http://127.0.0.1:7860"

const common = { agentRules: false, devIndicators: false }   // no AGENTS.md / CLAUDE.md written here by `next dev`

export default function config(phase) {
  if (phase !== PHASE_DEVELOPMENT_SERVER) return { ...common, output: "export" }
  return {
    ...common,
    async rewrites() {
      return [
        { source: "/ws", destination: `${EARS}/ws` },
        { source: "/voices", destination: `${EARS}/voices` },
        { source: "/classic", destination: `${EARS}/classic` },   // the classic page, with its meters and sliders
      ]
    },
  }
}
