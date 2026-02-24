# Output Design Patterns

Apply these patterns when a skill must produce repeatable, high-signal outputs.

## Strict Template Pattern

Use when consumers depend on exact section names or stable structure.

```markdown
# [Title]

## Summary
[Single paragraph]

## Findings
- [Finding + evidence]
- [Finding + evidence]

## Actions
1. [Actionable item]
2. [Actionable item]
```

When to use:
- API-like responses
- Compliance or audit outputs
- Any downstream parser-sensitive flow

## Guided Template Pattern

Use when consistency matters but adaptation should remain possible.

```markdown
# [Title]

## Summary
[Overview]

## Findings
[Adjust sections to fit context]

## Actions
[Prioritized recommendations]
```

When to use:
- Investigative analysis
- Planning tasks with variable scope
- User-facing synthesis where flexibility adds value

## Example-Driven Pattern

For style-sensitive outputs, include compact input/output examples.

```markdown
Example:
Input: Add OAuth login support
Output:
feat(auth): add OAuth login flow

Implement provider callback handling and token persistence.
```

Why it works:
- Reduces ambiguity better than abstract instructions
- Aligns tone/format quickly across similar tasks

## Pattern Selection Heuristic

- If format errors are expensive, choose strict template.
- If context varies heavily, choose guided template.
- If style/tone is the main risk, include examples.
