"use client"
// The full-screen particle animation (the template's), one per look; a change of look cross-fades.
import { AnimatePresence, motion } from "framer-motion"
import { ThinkingAnimation } from "@/components/animations/analyze"
import { AbsorptionAnimation } from "@/components/animations/think"
import { DeepAnimation } from "@/components/animations/deep"
import { TransferAnimation } from "@/components/animations/transfer"
import { IntenseAnimation } from "@/components/animations/intense"
import type { ShapeType } from "@/lib/shapes"

const ANIMATIONS = {
  think: AbsorptionAnimation,      // teal ripples from the centre
  analyze: ThinkingAnimation,      // purple grid, signals running through it
  deep: DeepAnimation,             // violet spiral
  transfer: TransferAnimation,     // amber waves
  intense: IntenseAnimation,       // red pulses
}
export type AnimationName = keyof typeof ANIMATIONS

export function Backdrop({ name, shape, dim }: { name: AnimationName; shape: ShapeType; dim: boolean }) {
  const Animation = ANIMATIONS[name]
  return (
    <div className="fixed inset-0 bg-[#05060a]" aria-hidden>
      <AnimatePresence initial={false}>
        <motion.div key={name} className="absolute inset-0" initial={{ opacity: 0 }}
                    animate={{ opacity: dim ? 0.35 : 1 }} exit={{ opacity: 0 }}
                    transition={{ duration: 0.7, ease: "easeInOut" }}>
          <Animation shape={shape} />
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
