import { PHASE_DEVELOPMENT_SERVER } from "next/constants.js"

// EARS (the web app, :7860) serves the WebSocket and the voice list. `npm run dev` runs this page on :3000 and passes
// those through to EARS (EARS_URL), so the page talks to one origin, as it does when EARS serves it. `npm run build`
// exports static files to out/, which EARS serves on its own port (app/ears/web.py).
const EARS = process.env.EARS_URL || "http://127.0.0.1:7860"

const common = { agentRules: false, devIndicators: false }   // no AGENTS.md / CLAUDE.md written here by `next dev`

export default function config(phase) {
  if (phase !== PHASE_DEVELOPMENT_SERVER) {
    return { ...common, output: "export", env: { NEXT_PUBLIC_CLASSIC: "/" } }
  }
  return {
    ...common,
    env: { NEXT_PUBLIC_CLASSIC: "/classic" },
    async rewrites() {
      return [
        { source: "/ws", destination: `${EARS}/ws` },
        { source: "/voices", destination: `${EARS}/voices` },
        { source: "/classic", destination: `${EARS}/` },        // the old page, with its meters and sliders
      ]
    },
  }
}
