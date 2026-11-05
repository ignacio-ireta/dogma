# Topic illustrations in hand-authored SVG

The selected Memphis Study Club theme uses original vector figures instead of the generated
production bitmap. Keep the vibrant coral, violet and cobalt palette against a cool background.
The user explicitly requested this replacement and subject-specific artwork on each topic page.

- Draw the people and subject motifs directly as SVG paths and shapes, without image generation
  or embedded raster assets. Keep the drawings editable in the source tree and offline.
- Welcome uses a general study scene. Today uses the active session's `topic_id`, with a general
  study scene for missing or unknown topics. Exercise pages use their route's actual topic ID.
- The 168 authored topics with exercises share a carefully assigned set of subject motifs.
  Reuse figures and related motifs across topics, with vectors, matrices, calculus, probability,
  SQL, language, networks and other subjects visually distinct.
- Exercise headers get a small illustration outside the reading/editor panels. On narrow screens
  it stays small and never squeezes the title. Preserve lesson typography, math, tables, focus,
  navigation, code execution, progress and all existing learning behavior.
- Artwork is decorative (`aria-hidden`, not focusable), static, and contains no learning claims.
  No new routes, dependencies, network assets or changes to learner data.

Verification: mapping coverage against authored packages; representative route switching and
null-topic fallback; desktop/mobile visual inspection; existing frontend and browser checks.
