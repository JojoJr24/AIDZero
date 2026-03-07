# Workflow Playbook

Use this reference when shaping the execution section of a skill.

## Linear Workflow Pattern

Best when the task has strict ordering and every step depends on the previous one.

```markdown
Execution flow:
1. Inspect input artifacts
2. Build or update intermediate mapping/config
3. Validate the mapping/config
4. Execute transformation
5. Verify output and report anomalies
```

## Decision Workflow Pattern

Best when the skill must branch according to request type or input state.

```markdown
Execution flow:
1. Classify request intent
   - New content path -> follow Creation Steps
   - Existing content path -> follow Editing Steps
2. Run branch-specific steps
3. Merge into one common validation/output section
```

## Hybrid Pattern

Use branching at the beginning, then converge to a common linear tail.

Recommended shape:
- Classification block
- Branch A steps
- Branch B steps
- Shared verification/output block

## Practical Guidance

- Keep step labels action-first (`Validate schema`, `Generate summary`).
- Avoid very large monolithic step lists; split by phase when needed.
- If failure handling matters, call out retry/fallback behavior directly in steps.
