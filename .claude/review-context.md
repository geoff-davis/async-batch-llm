# Review context: v0.25.0 release preparation

The executor and SQLite evidence changes have completed independent review.
Review release preparation for accurate metadata, packaging, and documentation;
report new evidence rather than reopening the accepted scope below.

## Accepted implementation and evidence

- D1 and D2 are merged through PRs #162 and #163. Private admission stages and
  attempt outcomes preserve public hooks, original exception identity, quota
  ownership, and cleanup order. Classification and cumulative failed usage are
  no longer transported through mutable exception attributes.
- Provider/custom usage stamps remain compatibility inputs. A hook that writes
  an explicit zero dictionary reports known zero; an unknown compatibility
  dictionary copied into a stamp loses its provenance. This limitation is
  documented in `docs/token-aware-admission.md`.
- Session E is reviewed in PR #164. Keep the SQLite backup timer and all three
  indexes. The historical CPython wait claim lacks an independently reproduced
  trigger. `docs/sqlite-maintenance-evidence.md` records evidence and limits.
- Public signatures, result schemas, artifact compatibility, replay eligibility,
  and cleanup contracts are unchanged. Durable historical decisions are in
  `docs/cleanup-lifecycle-contract.md`, `docs/results-and-artifacts.md`, and the
  v0.24/v0.25 migration guides.

## Release boundary

Session E must merge before the v0.25.0 release branch. Tagging and publication
require a separate maintainer go-ahead. No SQLite migration, new execution API,
provider-module split, or statistics-policy change is part of this release.

For new behavioral findings, follow `AGENTS.md`: reproduce against an isolated
base, preserve the review target, and exercise both artifact backends when
record eligibility or replay changes. A passing suite alone does not establish
coverage of a newly identified path.
