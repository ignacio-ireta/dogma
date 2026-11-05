# Memphis Study Club: selected production direction

The user selected Memphis Study Club with a quieter lesson workspace and requested more vibrant contrast and a slightly less yellow background. This follows the six proposals in `docs/design/2026-11-04-visual-proposals/README.md`; the selection authorizes applying the design to the existing app.

Use a near-neutral off-white canvas, white reading surfaces, deep navy text, saturated coral and violet decoration, and small cobalt accents. Distinct decorative tokens must not replace the established blue/input, orange/pending, green/success, and red/error meanings. Meet 4.5:1 for normal text and 3:1 for focus/control boundaries where required.

Bring the original Memphis illustration, bold outlines, offset shadows, and varied card surfaces to Today and Start here. Give the shared navigation a clear violet active state. Other overview pages inherit the palette and restrained geometric accents. Keep the lesson/editor panels white, their borders quiet, and mathematical typography intact.

Keep all routes, session actions, progress, hints, answers, code execution, keyboard controls, and learner data behavior. Preserve OpenDyslexic Bold for the interface, Inter for Markdown, KaTeX, and JetBrains Mono. Reuse the React/CSS architecture without new dependencies. All CSS color literals remain in tokens.css.

The original alternatives remain saved as design studies. This implementation uses the selected Memphis identity with a more vivid, cooler palette; it does not introduce a new learning flow. The existing feature branch contains earlier authorized typography/Markdown changes, which must remain intact.
