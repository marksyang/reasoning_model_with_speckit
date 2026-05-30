<!--
  SYNC IMPACT REPORT — v1.0.0
  ==========================
  Version change: 0.0.0 (new) → 1.0.0
  Modified principles: N/A (fresh creation)
  Added sections:
    - Core Principles (4 principles): Code Quality, Testing, UX Consistency, Performance
    - Development Workflow
    - Governance
  Removed sections: N/A
  Templates requiring updates:
    - plan-template.md: ✅ Constitution Check section references generic placeholder, compatible
    - spec-template.md: ✅ No constitution references found
    - tasks-template.md: ✅ No constitution references found
  Follow-up TODOs: None
-->

# Reasoning Model Project Constitution

## Core Principles

### I. Code Quality

All code MUST adhere to established code quality standards including [TODO](README.md), Clean Code principles, and relevant language-specific best practices (e.g., SOLID, DRY, KISS).

- Every function MUST have a single, well-defined responsibility.
- Variable and function names MUST convey intent — abbreviations and single-letter names are prohibited except in obvious loops (`i`, `j`).
- Functions MUST be short enough to read in one screen and MUST not exceed 30 lines unless justified.
- Constants MUST be extracted from hardcoded values with meaningful names.
- Comments MUST explain *why*, not *what*. Code that requires comments to be understood MUST be refactored.

**Why:** Consistent, readable code reduces bugs, accelerates onboarding, and enables confident refactoring.

### II. Testing

Tests are non-negotiable and MUST follow the established testing standards.

- Every new feature MUST have corresponding tests before it is merged.
- Unit tests MUST cover all public APIs and critical business logic.
- Edge cases MUST be tested alongside happy paths, especially input validation and error conditions.
- Integration tests MUST cover cross-module and cross-service interactions.
- Tests MUST be independent — no hidden state dependencies, no order-dependent execution.
- Every bug fix MUST include a regression test.
- Test code MUST follow the same quality standards as production code.
- Code coverage targets MUST be maintained — no pull request that reduces coverage below the project threshold.

**Why:** Tests provide the safety net for refactoring, prevent regression, and serve as living documentation of expected behavior.

### III. User Experience Consistency

Every user-facing interface MUST maintain consistent and intuitive experience standards.

- UI components MUST follow established design patterns and component conventions across the project.
- Error messages MUST be specific, actionable, and written in a consistent tone.
- Error states MUST provide recovery paths — never dead ends.
- Loading states, empty states, and success states MUST be designed for every user interaction.
- Accessibility standards MUST be met — color contrast, keyboard navigation, ARIA labels.
- User feedback MUST be immediate and visible — operations MUST communicate their progress.
- Input fields MUST validate in real time and display errors inline.

**Why:** Consistency reduces cognitive load, accelerates user proficiency, and builds trust in the product.

### IV. Performance

Performance is a feature, not an afterthought.

- Every component MUST meet defined performance budgets (e.g., render time, memory, network).
- Expensive operations MUST be memoized, debounced, or batched appropriately.
- Unnecessary re-renders MUST be eliminated through shouldComponentUpdate or equivalent mechanisms.
- User interactions MUST respond within 100ms for perceived instantaneity, under 500ms for acceptable latency.
- Bundle size MUST be monitored — tree-shaking, lazy loading, and code splitting MUST be used where applicable.
- Database queries MUST be optimized — N+1 queries MUST be eliminated through proper joining or batching.

**Why:** Users abandon slow interfaces. Proactive performance management prevents costly re-architecture and delivers measurable business value.

## Development Workflow

### Coding Standards

- All code MUST follow the project's [clean code guide](CLAUDE.md) — read it before starting implementation.
- PRs MUST be small and focused — ideally under 400 lines of code.
- Every PR MUST include test changes alongside implementation changes.
- PR descriptions MUST explain *why* changes were made, not *what* was changed.
- Dependencies MUST be reviewed for security vulnerabilities before adding.

### Review Process

- Every PR MUST be reviewed by at least one team member before merging.
- Reviewers MUST verify: correctness, test coverage, performance impact, and UX consistency.
- Rejections MUST provide clear, actionable feedback.

## Governance

This constitution supersedes all other development practices and conventions within this project.

**Amendments:**
1. Propose changes via pull request to this document.
2. Changes require team consensus and documented rationale.
3. Backward-incompatible principle changes require a migration plan.

**Compliance:**
- All PRs and reviews MUST verify alignment with these principles.
- Violations MUST be logged, analyzed, and addressed in retro.
- Complexity beyond this constitution MUST be justified with concrete examples.
- Use the [code quality guide](CLAUDE.md) for day-to-day development decisions.

**Versioning:** Changes to this constitution follow semantic versioning (MAJOR.MINOR.PATCH).

**Version**: 1.0.0 | **Ratified**: 2026-05-30 | **Last Amended**: 2026-05-30