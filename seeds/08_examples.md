# Patterns that work

Use a time-based animation loop so behaviour is consistent regardless of frame rate:

```js
let last = performance.now();
function tick(now) { const dt = Math.min((now - last) / 1000, 0.05); last = now; update(dt); requestAnimationFrame(tick); }
requestAnimationFrame(tick);
```

Attach pointer handlers for the whole drag lifecycle, tracking velocity from the last few pointer positions so a release can inherit momentum:

```js
el.addEventListener('mousedown', start); window.addEventListener('mousemove', move); window.addEventListener('mouseup', end);
```

Keep objects inside their container by clamping position and reflecting velocity when a boundary is crossed. Guard DOM lookups and register `resize` listeners on `window`.
