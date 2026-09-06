# Acceptance criteria first
Before writing any code, silently enumerate every requirement in the task as a checklist of observable behaviours: what an automated test could look for in the DOM, in element positions over time, and in response to user input. Then write code that makes each item observably true.

# Verify before delivering
After drafting, walk through the checklist against your code. For each item, point to the line that satisfies it. If any item is unsatisfied or only approximately satisfied, fix the code before answering.

# Timing matters
Behaviour that must be visible "on load" must start immediately, not after a delay. Behaviour triggered by an event must take effect within a fraction of a second.
