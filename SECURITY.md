# Security policy

## Supported versions

Security fixes go into the latest release. Before 1.0, only the most recent minor
version is supported; upgrade to receive fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Report a vulnerability](https://github.com/geoff-davis/async-batch-llm/security/advisories/new).
Don't open a public issue for a suspected vulnerability.

Include the affected version, a minimal reproduction if you have one, and the impact
you expect. This is a small, maintainer-run project: expect an acknowledgement within
a week. Fixes are released as a new patch version with a GitHub security advisory.

## Scope

async-batch-llm runs inside your process and calls the provider SDKs you configure.
Reports are most useful for problems in the library itself, for example:

- artifact or result files written with content the privacy settings say is omitted
  (prompts and context are excluded by default; see
  [Results and artifacts](docs/results-and-artifacts.md));
- replay of a stored result for a request it doesn't match;
- API keys or other credentials appearing in logs, results or artifacts.

Vulnerabilities in a provider SDK or in a dependency should be reported to that
project. Artifact stores and serialized results are read as JSON or SQLite and never
executed, but any `output_decoder` or `context_decoder` you supply runs on their
contents, and a replayed result is returned as if the provider had produced it. Only
resume from files you trust.
