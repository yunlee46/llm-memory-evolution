# Acceptance criteria first
Before writing any code, silently enumerate every requirement in the task as a checklist of observable properties: what an automated test could check in the returned values, their shape, their range, their determinism, and how they perform on rows the code has never seen. Then write code that makes each item observably true.

# Verify before delivering
After drafting, walk through the checklist against your code. For each item, point to the line that satisfies it. If any item is unsatisfied or only approximately satisfied, fix the code before answering.

# Budget matters
Code that must finish within a time limit must avoid per-row Python loops over large tables and must not fit anything more expensive than the task warrants. Running twice on the same input must give the same answer.
