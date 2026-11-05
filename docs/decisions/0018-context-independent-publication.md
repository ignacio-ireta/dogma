# 0018. Context-independent publication and privacy checks

Date: 2026-10-29 · Status: accepted

## Context

Published requirements and examples should describe the product and technical skills without
profiling an individual. A content-only privacy check misses filenames, package metadata,
vendored fixtures, staged blobs and generated browser assets. Git history is a separate source
of earlier wording and author metadata.

## Decision

Use general learning-design principles and synthetic examples. Import reviewed reference
lessons through an explicit allowlist, without naming excluded personal source files.

Require local, ignored privacy rules. Check all publication files and newly built assets;
inspect common escaped text and UTF-16 as well as UTF-8. Reject publication symlinks and
force-tracked private state without reading private file bodies. Pre-commit additionally checks
actual index blobs with `--staged`, including files deleted only from disk.

Retain required third-party attribution. Narrow exemptions are pinned to complete license
checksums and exact opaque parser-table string checksums; modified copies and surrounding code
remain checked. Diagnostics identify rule numbers and redact matching filenames.

Provide history-free source snapshots for publication without altering existing Git history.
Exclude private state, local rules, dependency environments and agent artifacts. Audit an
exported tree independently of Git; normalize archive timestamps and omit filesystem ownership.

## Consequences

Development privacy checks require a nonempty local rules file. A working-tree audit does not
certify staged bytes unless `--staged` is used, and neither mode certifies earlier commits.
The scanner checks text and filenames rather than claiming arbitrary binary or adversarial
encoding analysis. Required public scholarly and license attribution is preserved.

## Alternatives

- Blanket removal of names or matching letter sequences would damage scholarly attribution,
  licenses and generated parser data; use reviewed checksum-pinned exceptions instead.
- Auditing before the browser build could miss newly generated references; audit after it.
- Rewriting existing history changes branch and commit identities; a clean export provides a
  publication artifact while preserving local development history.
