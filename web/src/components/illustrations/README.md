# Study Club vector drawings

These are original, hand-authored SVG paths and geometric shapes, drawn directly in React.
They replace the generated bitmap formerly used in the production site. There is no image
generation, tracing, embedded bitmap, external asset, font-dependent lettering or animation.

- `GuideFigure.tsx`: the coral-jacket presenter with a long stride and pointing hand.
- `ReaderFigure.tsx`: the violet-shirt reader sitting on a stool with an open book.
- `StudyIllustration.tsx`: the full study scene and the compact exercise-header composition.
- `*Motifs.tsx`: 20 subject motifs plus a general open-book motif, drawn in a 180 × 120 space.
- `TopicMotif.tsx`: the complete, type-checked motif collection.
- `../../lib/topicIllustration.ts`: deliberate assignments for authored topics. Related topics
  share a motif; every existing exercise page has an assignment. Unknown IDs use the book scene.
- `../../styles/illustrations.css` and `tokens.css`: shared palette and stroke treatment.

Welcome uses the book scene; Today uses the current session's topic; each exercise header uses
its actual topic ID. These are decorative subject cues, not instructional diagrams. The lesson
retains its own accessible prose, formulas and tables. All SVG roots are hidden from assistive
technology and have no focus stops or IDs that could collide when repeated.

When adding a topic package, choose its scene in the mapping. The inventory test checks actual
authored content so a forgotten assignment cannot silently ship as generic artwork.

The older raster comparison in `docs/design/2026-11-04-visual-proposals/` is a historical design
study, not a production dependency.
