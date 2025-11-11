# Migration Guides

This directory contains migration guides for upgrading between major versions of batch-llm.

## Current Version: 0.3.0

**Latest stable release:** v0.3.0 (2025-01-10)

---

## Available Migration Guides

### 🔄 [v0.2 → v0.3 Migration Guide](./MIGRATION_V0_3.md)

**Status:** ✅ Complete
**Breaking Changes:** None (100% backward compatible)
**Release Date:** 2025-01-10

**Key Features:**

- RetryState for multi-stage retry patterns
- GeminiResponse for safety ratings access
- Cache tagging for precise cache matching

**Who should read this:**

- Everyone upgrading from v0.2.x to v0.3.0
- Users wanting to adopt new retry state features
- Users needing content moderation with safety ratings
- Multi-tenant applications needing cache isolation

**Migration complexity:** 🟢 **Easy** - All new features are opt-in

---

### 🔄 [v0.1 → v0.2 Migration Guide](./MIGRATION_V0_2.md)

**Status:** ✅ Complete
**Breaking Changes:** None (100% backward compatible)
**Release Date:** 2024-10-22

**Key Features:**

- Shared strategy lifecycle (70-90% cost savings)
- Automatic cache renewal for GeminiCachedStrategy
- Enhanced cache management and reuse

**Who should read this:**

- Users upgrading from v0.1.x to v0.2.0
- Users of GeminiCachedStrategy wanting cost optimization
- Anyone using multiple work items with the same strategy

**Migration complexity:** 🟢 **Easy** - All new features are opt-in

---

### 🔄 [v0.0 → v0.1 Migration Guide (Strategy Pattern)](./MIGRATION_V3.md)

**Status:** ✅ Complete
**Breaking Changes:** Yes - Major API refactor
**Release Date:** 2024-10-15

**Key Changes:**

- Replaced `agent=`, `agent_factory=`, `direct_call=` with unified `strategy=` parameter
- Introduced `LLMCallStrategy` abstract base class
- Framework-level timeout enforcement with `asyncio.wait_for()`
- Built-in strategies: `PydanticAIStrategy`, `GeminiStrategy`, `GeminiCachedStrategy`

**Who should read this:**

- Users upgrading from v0.0.x to v0.1.0+
- Anyone using the old agent-based API
- Users migrating from PydanticAI agent parameters

**Migration complexity:** 🟡 **Medium** - Requires code changes to use new strategy API

**Note:** This file is named `MIGRATION_V3.md` for historical reasons (internal versioning), but refers to the v0.1.0 release.

---

## Quick Reference: What Changed When?

| Version | Release Date | Breaking Changes | Key Features |
|---------|--------------|------------------|--------------|
| **v0.3.0** | 2025-01-10 | None | RetryState, GeminiResponse, Cache tags |
| **v0.2.0** | 2024-10-22 | None | Shared strategy lifecycle, Auto cache renewal |
| **v0.1.0** | 2024-10-15 | **Yes** | Strategy pattern refactor, Unified API |
| **v0.0.x** | 2024-10-01 | N/A | Initial releases (agent-based API) |

---

## Migration Path Finder

### "I'm on v0.0.x and want to upgrade to latest"

1. Read [v0.0 → v0.1 Migration Guide](./MIGRATION_V3.md) **(Required - breaking changes)**
2. Read [v0.1 → v0.2 Migration Guide](./MIGRATION_V0_2.md) (Optional - no breaking changes)
3. Read [v0.2 → v0.3 Migration Guide](./MIGRATION_V0_3.md) (Optional - no breaking changes)

**Estimated effort:** 2-4 hours (mostly for v0.1 strategy pattern refactor)

---

### "I'm on v0.1.x and want to upgrade to latest"

1. Read [v0.1 → v0.2 Migration Guide](./MIGRATION_V0_2.md) (Optional - no breaking changes)
2. Read [v0.2 → v0.3 Migration Guide](./MIGRATION_V0_3.md) (Optional - no breaking changes)

**Estimated effort:** 30 minutes - 1 hour (only if adopting new features)

---

### "I'm on v0.2.x and want to upgrade to v0.3.0"

1. Read [v0.2 → v0.3 Migration Guide](./MIGRATION_V0_3.md) (Optional - no breaking changes)

**Estimated effort:** 15-30 minutes (only if adopting new features)

---

## Deprecation Policy

batch-llm follows semantic versioning:

- **Patch releases (0.3.x):** Bug fixes only, no breaking changes, no deprecations
- **Minor releases (0.x.0):** New features, may deprecate old features with warnings, no breaking changes
- **Major releases (x.0.0):** May remove deprecated features, may have breaking changes

**Deprecation timeline:**

1. Feature marked as deprecated in minor release (with warning)
2. Documented in CHANGELOG and migration guide
3. Removed in next major release (minimum 6 months notice)

---

## Backward Compatibility Guarantee

**v0.1.0+**: We maintain backward compatibility across minor versions:

- v0.1.x → v0.2.0: ✅ No breaking changes
- v0.2.x → v0.3.0: ✅ No breaking changes
- v0.3.x → v0.4.0: ✅ No breaking changes expected

**v0.0.x**: Legacy versions, no backward compatibility guarantee.

---

## Getting Help

### If you encounter migration issues

1. **Check the migration guide** for your version in this directory
2. **Search GitHub Issues**: <https://github.com/anthropics/batch-llm/issues>
3. **Open a new issue**: Include:
   - Source version (e.g., "v0.1.0")
   - Target version (e.g., "v0.3.0")
   - Error message or unexpected behavior
   - Minimal code example

### Common Migration Questions

**Q: Do I need to migrate immediately after a new release?**

A: No. Minor releases (0.x.0) are backward compatible. You can upgrade at your own pace. Only major releases
(x.0.0) may require migration.

**Q: Can I skip versions when upgrading?**

A: Yes, if there are no breaking changes. However, we recommend reading all migration guides between your
version and the target version to learn about new features.

**Q: What if I'm using a deprecated feature?**

A: Deprecated features will show warnings but continue to work until the next major release. Plan to migrate
during the deprecation period.

**Q: How do I know if a release has breaking changes?**

A: Check the CHANGELOG.md file. Breaking changes are clearly marked with a warning symbol at the top of each
release section.

---

## Version History Reference

For complete details on each release, see [CHANGELOG.md](../CHANGELOG.md).

For current development roadmap, see [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md).

---

## Contributing Migration Guides

When releasing a new version with changes that affect users:

1. Create `docs/MIGRATION_V0_X.md` following the existing template
2. Update this index file with the new guide
3. Mark breaking changes clearly with ⚠️ symbols
4. Provide before/after code examples
5. Estimate migration complexity (Easy/Medium/Hard)
6. Include links to relevant issues and documentation

**Template structure:**

- Overview and key changes
- Breaking changes (if any)
- Step-by-step migration instructions
- Before/after code examples
- Testing migration checklist
- Common pitfalls and solutions
- Additional resources

---

**Note:** Last updated: 2025-01-10
