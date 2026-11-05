# dogma: six visual directions

Design exploration requested on 4 November 2026: up to five proposals informed by learning and coding websites, plus one Corporate Memphis direction. These are reviewable mockups, not changes to the running application.

Open [index.html](index.html) in a browser. It works offline and provides:

- Six distinct Today layouts, with a comparison gallery.
- A lesson/editor view for each direction, including real KaTeX formulas and a table.
- Desktop and phone previews.
- Palettes, rationale, trade-offs, and reference notes.

The figures and progress are sample data; durations are estimates, not targets. Navigation between Today and the lesson is simulated. The workspace includes interactive Statement, Lesson, Answer, and Hints panels. Run and Submit only display a preview message; they do not execute code or change learner data. Map, Review, Mock, Progress, and Start here show their navigation appearance only.

## Product brief and existing workflow

The follow-up brief describes Dogma as free, accessible, curiosity-driven, laidback, youthful, intellectually credible, and non-coercive. Its desired response is “This looks interesting. I want to explore it.” These proposals add visual identity through composition, color, illustration, and hierarchy while keeping the current learning method.

The codebase uses React and TypeScript with Vite, shared CSS tokens, a navigation rail, Markdown/KaTeX lesson content, and a CodeMirror workbench backed by the local API. Inspection covered `App.tsx`, `AppShell.tsx`, `NavRail.tsx`, `ProblemPage.tsx`, `HintsPanel.tsx`, `answer/AnswerPanel.tsx`, and the typography tokens. The prototypes are separate HTML/CSS/JavaScript documents under this directory; they do not replace those components.

| Requirement | Representation in the proposals |
| --- | --- |
| Free and accessible | No account or payment barrier; bundled assets work offline; readable prose/math, keyboard controls, scrollable tables, and responsive layouts |
| Curiosity and relaxed progress | Invitations to explore ideas, optional guidance, estimated durations, and a record of explored sessions without a weekly quota or streak |
| Youthful and credible | Six distinct compositions, including original Memphis artwork, balanced with consistent mathematical and code typography |
| Preserve navigation | Today, Problem, Map, Review, Mock, Progress, Start here remain in their existing order |
| Introduction, theory, examples | Statement and Lesson panels show the same vector-projection topic, real equations, a formula table, and a worked example |
| Optional hints and solutions | Hints offers Concept, Approach, and Code shape; a sample full walkthrough is explicitly requested and confirmed |
| Practice and execution | The familiar lesson/editor/results arrangement and Run/Submit controls remain visible; execution is simulated in the study |
| Progress without pressure | Neutral session history and topic context; no mandatory daily cadence, countdown, ranking, or notification system |
| Visual exploration only | All six directions stay reviewable alternatives; a recommendation is not a selected production redesign |

**Answer retains its existing meaning:** an interview question answered in the learner’s own words with self-assessment. It is not a solution-reveal tab. The full walkthrough remains under Hints. The preview represents an already eligible sample after two failed submits and requires confirmation; the real API’s eligibility rules remain untouched. Form entries exist only for the current preview and reset when it reloads.

## The directions

| Direction | Visual approach | Best fit | Main trade-off |
| --- | --- | --- | --- |
| 01 Practice Studio | Warm white, forest green, compact navigation, a prominent next action | A focused evolution of the current app | Familiar rather than strongly distinctive |
| 02 Learning Atlas | Teal, mint, connected topic nodes, visible progression | Seeing how a study session fits the curriculum | Needs care as the map grows |
| 03 Research Notebook | Paper tones, ink blue, chapter numbers, editorial rules and diagrams | Reading, mathematical explanation, reflection | Less content density |
| 04 Precision Console | Cool neutrals, blue accents, aligned rows and compact statuses | Frequent practice and rapid scanning | Can feel dense |
| 05 Geometry Lab | Lavender, graph-paper accents, vector geometry and topic symbols | A distinctive ML learning identity | Requires a consistent, accurate diagram system |
| 06 Memphis Study Club | Cream, navy, coral, lavender, expressive figures and chunky cards | A warm, recognizable identity matching the requested style | Requires a small coherent illustration set |

Recommendation: **Memphis Study Club with Practice Studio's workspace discipline**. Use illustration on Today and curriculum/progress surfaces; let the lesson and editor remain visually calm. This is a recommendation, not an assumed selection.

All directions preserve OpenDyslexic Bold for interface labels, Inter for reading, KaTeX for mathematics, and JetBrains Mono for code. The shared lesson preview makes the typography directly comparable. Geometry Lab's diagram is static; interactive teaching diagrams would be an additional feature.

## Research and access limits

Reviewed public sources on 4 November 2026. Deep-ML's homepage was inspected in a live browser. HackerRank, LeetCode, Codeforces, and DataCamp blocked the automated browser with HTTP 403 responses. Their accessible public page text and official documentation informed the structural analysis; their current authenticated interfaces were not inspected. LeetCode Explore's official screenshots are historical (2018 images, article updated 2022).

| Reference | Observed pattern used in these proposals |
| --- | --- |
| [HackerRank practice](https://www.hackerrank.com/domains/algorithms) | Status, skills, and difficulty filters; explicit solve actions |
| [HackerRank coding workspace documentation](https://candidatesupport.hackerrank.com/articles/9623161883-answer-coding-questions) | Adjustable question/editor layouts; Run/Submit hierarchy |
| [LeetCode problem library](https://leetcode.com/problemset/) | Topic organization and question search |
| [LeetCode Explore documentation](https://support.leetcode.com/hc/en-us/articles/360013578114-How-to-use-Explore) | Chapter navigation and completion markers, as a historical structural reference |
| [Deep-ML](https://www.deep-ml.com/) | Technical identity, concrete code example, routes to practice and curriculum |
| [Codeforces problemset](https://codeforces.com/problemset) | Dense problem rows with tags, difficulty filters, and submission counts |
| [DataCamp learning track](https://www.datacamp.com/tracks/associate-data-scientist-in-python) | A guided progression from foundations to practical work |

The proposals are original interpretations. No competitor artwork or UI screenshots are redistributed in the mockups.

## Files and verification

`concepts.js` contains rationale and references; `preview.js`, `preview.css`, and `reading.css` contain the six mockups and their shared reading/interaction treatment. `board.js` and `board.css` power the comparison controls. Fonts and KaTeX are local under `assets/`; [font licensing notes](assets/fonts/README.md) identify the bundled licenses, including the repository's canonical OpenDyslexic license.

The original Corporate Memphis illustration is [assets/memphis-study.webp](assets/memphis-study.webp); its generation method and prompt are recorded in [ARTWORK.md](ARTWORK.md).

Final browser QA covered all twelve concept/screen combinations at 1160, 390, and 320 pixels wide, after fonts loaded: all 36 cases fit without page-level horizontal overflow. Keyboard tab navigation, answer draft retention, optional hints, walkthrough confirmation, and simulated Run/Submit feedback passed. The checked comparison-board labels measured 5.8:1 contrast. No JavaScript errors or failed asset requests were observed. Six refreshed Today screenshots are saved under `assets/previews/` for the comparison gallery. Selected Today, Lesson, and Answer layouts were also inspected visually.

The displayed walkthrough was checked against the actual vector-projection exercise: normal, tiny, and huge directions; empty, mismatched, and invalid shapes; zero direction; float64 output; and unchanged inputs. These checks validate the sample content, not a new code-execution feature.
