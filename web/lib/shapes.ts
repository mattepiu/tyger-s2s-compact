// The particles' shapes (the template's shape tabs): each character draws its own.
export type ShapeType = "circle" | "triangle" | "square" | "diamond"

export const SHAPES: ShapeType[] = ["circle", "triangle", "square", "diamond"]
export const GLYPHS: Record<ShapeType, string> = { circle: "●", triangle: "▲", square: "■", diamond: "◆" }
