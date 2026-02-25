---
description: 'AgentX UX methodology -- structured user-centered design phases for the UX Designer agent.'
applyTo: '**/ux/**,**/UX-*,**/prototypes/**,**/wireframe*'
---

# AgentX UX Methodology

AgentX uses a **5-phase UX methodology** (Empathize, Define, Ideate, Prototype, Validate) to guide the UX Designer agent through structured, user-centered design work.

---

## UX Design Phases

### Phase 1: Empathize (Research)

**Goal**: Understand users, their context, and their pain points.

**MUST** perform before any design work:
1. Read the PRD thoroughly -- extract user stories, personas, and acceptance criteria
2. Research existing patterns in the codebase (`semantic_search`, `grep_search`)
3. Identify target user segments and their goals
4. Document assumptions to validate

**Deliverables**:
- User personas (demographics, goals, frustrations, behaviors)
- Empathy map (Says, Thinks, Does, Feels)
- Current user journey (if existing product)

**Token budget**: Keep research notes under 2000 words. Summarize, do not copy.

---

### Phase 2: Define (Problem Framing)

**Goal**: Synthesize research into clear problem statements.

**MUST** produce:
1. **Problem statement**: "Users need a way to [goal] because [insight]"
2. **How Might We (HMW) questions**: 3-5 open design challenges
3. **Success metrics**: Measurable outcomes tied to PRD acceptance criteria
4. **Constraints**: Technical limits from ADR/specs, accessibility requirements, platform targets

**Format**:
```markdown
## Problem Statement
[User type] needs [capability] because [reason/insight].

## How Might We
- HMW reduce the time to [task]?
- HMW make [feature] accessible to [user group]?
- HMW provide feedback when [action] completes?

## Success Metrics
- Task completion rate >= X%
- Time to complete [flow] < Y seconds
- Accessibility: WCAG 2.1 AA pass
```

---

### Phase 3: Ideate (Exploration)

**Goal**: Generate multiple design approaches before committing to one.

**MUST** explore at least 2 distinct approaches:
1. Sketch 2-3 alternative layouts (ASCII wireframes in markdown)
2. Compare trade-offs (simplicity vs power, familiar vs novel)
3. Select the approach with rationale documented

**MUST NOT**:
- Jump to a single solution without exploring alternatives
- Copy UI patterns without adapting to the project's design system

**Format for alternatives**:
```markdown
## Approach A: [Name]
- Layout: [ASCII wireframe]
- Pros: [list]
- Cons: [list]

## Approach B: [Name]
- Layout: [ASCII wireframe]
- Pros: [list]
- Cons: [list]

## Selected: Approach [X]
Rationale: [why this approach best serves users and constraints]
```

---

### Phase 4: Prototype (Build)

**Goal**: Create production-ready HTML/CSS prototypes.

**MUST** create:
1. **Lo-fi wireframes** in UX spec (ASCII art or simple diagrams)
2. **HTML/CSS prototypes** at `docs/ux/prototypes/` -- MANDATORY
3. **Component specs** (states, variants, responsive breakpoints)

**Prototype requirements**:
- Semantic HTML5 markup
- Clean CSS (BEM naming or CSS custom properties)
- Responsive: mobile (320px), tablet (768px), desktop (1024px+)
- Interactive states: hover, focus, active, disabled, error, loading
- WCAG 2.1 AA: color contrast >= 4.5:1, focus indicators, aria labels
- No external dependencies (vanilla HTML/CSS/JS only)

**SHOULD** include:
- Dark/light theme support via CSS custom properties
- Keyboard navigation for all interactive elements
- Screen reader testing notes

---

### Phase 5: Validate (Review)

**Goal**: Verify the design meets user needs and technical constraints.

**MUST** check before handoff:

| Check | Criteria | Required |
|-------|----------|----------|
| Completeness | All user stories with `needs:ux` addressed | YES |
| User flows | Happy path + error states + edge cases | YES |
| Accessibility | WCAG 2.1 AA (contrast, focus, aria, keyboard) | YES |
| Responsive | Mobile, tablet, desktop variants | YES |
| Prototypes | HTML/CSS files exist in `docs/ux/prototypes/` | YES |
| Consistency | Follows existing design system/brand | YES |
| Clarity | Engineer can build from spec without questions | YES |

**Validation command**:
```bash
./.github/scripts/validate-handoff.sh <issue_number> ux
```

---

## Phase Transitions

```
Empathize -> Define -> Ideate -> Prototype -> Validate -> Handoff
   |           |         |          |            |
   v           v         v          v            v
 Personas   Problem    2+ Ideas   HTML/CSS    Checklist
 Journey    HMW Qs    Selection  Prototypes   WCAG Pass
```

**Context preservation**: Each phase builds on the previous. Do NOT clear context between DT phases -- the UX agent maintains full context throughout.

**Handoff to Architect**: After Validate phase, context SHOULD be cleared before Architect begins (prevents design assumptions from leaking into technical decisions).

---

## Session Persistence

If the UX Designer agent is interrupted mid-design:
1. Save current phase and progress in `docs/progress/ISSUE-{id}-log.md`
2. On resume, read the progress log to recover state
3. Continue from the last completed phase

**Progress log format**:
```markdown
## DT Progress - Issue #{id}
- Phase: Ideate (in progress)
- Completed: Empathize, Define
- Current: Exploring Approach B layout
- Next: Finalize wireframes, begin prototyping
```

---

## Integration with AgentX Workflow

| AgentX Phase | DT Phase | Deliverable |
|-------------|----------|-------------|
| PM completes PRD | -- | PRD at `docs/prd/PRD-{id}.md` |
| UX starts | Empathize | Personas, user journey |
| UX continues | Define | Problem statement, HMW |
| UX continues | Ideate | Alternative layouts, selection |
| UX continues | Prototype | HTML/CSS at `docs/ux/prototypes/` |
| UX completes | Validate | Self-review checklist passed |
| Architect starts | -- | Reads UX spec, NOT DT notes |

---

**Version**: 1.0
**Last Updated**: February 21, 2026
