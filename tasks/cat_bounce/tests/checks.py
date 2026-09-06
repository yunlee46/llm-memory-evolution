"""Playwright checks for the cat-bounce task.

Exposes run(browser, url, cfg) -> list[CheckResult]. Two fresh page loads:
  A: load, static structure, gravity/bounce/bounds sampling, rain, resize.
  B: load, let cats settle, drag test, slow vs fast throw comparison.
Every check is isolated with try/except so one failure cannot mask the others.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    weight: float
    passed: bool
    detail: str = ""


SETTLE_S = 4.0


def _boxes(page):
    """Bounding boxes of all .cat elements: list of dicts with x,y,w,h (viewport px)."""
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('.cat')).map(el => {
             const r = el.getBoundingClientRect();
             return {x: r.left, y: r.top, w: r.width, h: r.height};
           })"""
    )


def _stage_box(page):
    return page.evaluate(
        """() => { const s = document.getElementById('stage'); if (!s) return null;
                  const r = s.getBoundingClientRect();
                  return {x: r.left, y: r.top, w: r.width, h: r.height}; }"""
    )


def _sample_ys(page, seconds: float, step_s: float = 0.05):
    """Per-cat y trajectories (indexed by cat order at start). Cats added mid-way are ignored."""
    n = len(_boxes(page))
    traj = [[] for _ in range(n)]
    end = time.time() + seconds
    while time.time() < end:
        bs = _boxes(page)
        for i in range(min(n, len(bs))):
            traj[i].append(bs[i]["y"])
        time.sleep(step_s)
    return traj


def _has_bounce(ys, min_rise=10.0):
    """True if the trajectory goes down, hits a local minimum in height (max y), then rises."""
    if len(ys) < 5:
        return False
    peak_y = ys[0]
    descended = False
    for y in ys[1:]:
        if y > peak_y + 1:
            peak_y = y
            descended = True
        elif descended and peak_y - y >= min_rise:
            return True
    return False


def _new_page(browser, url, cfg, errors, external):
    vp = cfg.get("viewport", {"width": 1000, "height": 700})
    ctx = browser.new_context(viewport=vp)
    page = ctx.new_page()

    def on_route(route):
        if route.request.url.startswith(("file:", "data:", "blob:", "about:")):
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    page.route("**/*", on_route)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
    page.goto(url, wait_until="load", timeout=15000)
    return ctx, page


def _guard(results, name, weight, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {str(e)[:120]}"
    results.append(CheckResult(name, weight, bool(ok), detail))


def run(browser, url, cfg) -> list[CheckResult]:
    results: list[CheckResult] = []
    errors: list[str] = []
    external: list[str] = []

    # ---------------------------------------------------------------- page A
    ctx, page = _new_page(browser, url, cfg, errors, external)
    try:
        _guard(results, "title_mentions_cat", 1, lambda: ("cat" in page.title().lower(), page.title()[:60]))

        def stage_and_cats():
            sb = _stage_box(page)
            n = len(_boxes(page))
            return sb is not None and n >= 3, f"stage={'yes' if sb else 'no'} cats={n}"
        _guard(results, "stage_and_initial_cats", 2, stage_and_cats)

        def accessible():
            bad = page.evaluate(
                """() => Array.from(document.querySelectorAll('.cat'))
                        .filter(el => !(el.getAttribute('alt') || el.getAttribute('aria-label'))).length"""
            )
            n = len(_boxes(page))
            return n > 0 and bad == 0, f"{bad}/{n} cats missing alt/aria-label"
        _guard(results, "cats_accessible", 1, accessible)

        # gravity + bounce: sample from shortly after load
        page.wait_for_timeout(100)
        traj = _sample_ys(page, 3.0)

        def _falls(ys):
            """Three consecutive samples each moving down by >= 3px = under gravity."""
            run = 0
            for a, b in zip(ys, ys[1:]):
                run = run + 1 if b > a + 3 else 0
                if run >= 3:
                    return True
            return False

        fell_initial = sum(1 for ys in traj if _falls(ys))

        def cats_bounce():
            n = sum(1 for ys in traj if _has_bounce(ys))
            return n >= 1, f"{n}/{len(traj)} cats showed a floor bounce"
        _guard(results, "cats_bounce", 3, cats_bounce)

        # let things settle, then bounds
        page.wait_for_timeout(int((SETTLE_S - 3.1) * 1000))

        def in_bounds():
            sb = _stage_box(page)
            if not sb:
                return False, "no #stage"
            bs = _boxes(page)
            tol = 5
            out = [b for b in bs if b["x"] < sb["x"] - tol or b["y"] < sb["y"] - tol
                   or b["x"] + b["w"] > sb["x"] + sb["w"] + tol or b["y"] + b["h"] > sb["y"] + sb["h"] + tol]
            moved_any = any(len(ys) > 2 and max(ys) - min(ys) > 5 for ys in traj)
            if not moved_any:
                return False, "cats never moved, bounds check not earned"
            return len(bs) > 0 and not out, f"{len(out)}/{len(bs)} cats outside stage after {SETTLE_S}s"
        _guard(results, "cats_stay_in_bounds", 2, in_bounds)

        def rain():
            before = len(_boxes(page))
            btn = page.locator("#make-it-rain")
            if btn.count() == 0:
                return False, "no #make-it-rain"
            btn.first.click(timeout=3000)
            page.wait_for_timeout(150)
            # new cats spawn at the top: sample them briefly to observe gravity
            rain_traj = _sample_ys(page, 0.6)[before:]
            rain_fell[0] = sum(1 for ys in rain_traj if _falls(ys))
            page.wait_for_timeout(300)
            after = len(_boxes(page))
            return after - before >= 5, f"cats {before} -> {after}"
        rain_fell = [0]
        _guard(results, "rain_adds_cats", 2, rain)

        def cats_fall():
            ok = fell_initial >= 1 or rain_fell[0] >= 1
            return ok, f"initial cats falling: {fell_initial}/{len(traj)}, rained cats falling: {rain_fell[0]}"
        _guard(results, "cats_fall", 2, cats_fall)

        def bg_resize():
            def bg():
                return page.evaluate(
                    """() => { const b = getComputedStyle(document.body).backgroundColor;
                               const s = document.getElementById('stage');
                               const h = getComputedStyle(document.documentElement).backgroundColor;
                               return b + '|' + (s ? getComputedStyle(s).backgroundColor : '') + '|' + h; }"""
                )
            a = bg()
            page.set_viewport_size({"width": 900, "height": 650})
            page.wait_for_timeout(300)
            b = bg()
            return a != b, f"{a} -> {b}"
        _guard(results, "bg_changes_on_resize", 1, bg_resize)

        # loading/errors collected across page A's whole life
        _guard(results, "loads_without_errors", 2, lambda: (not errors, "; ".join(errors)[:160]))
        _guard(results, "no_external_requests", 1, lambda: (not external, "; ".join(external)[:160]))
    finally:
        ctx.close()

    # ---------------------------------------------------------------- page B: interaction
    errs_b: list[str] = []
    ctx, page = _new_page(browser, url, cfg, errs_b, [])
    try:
        page.wait_for_timeout(int(SETTLE_S * 1000))

        def pick_cat():
            """Choose a resting cat that is actually on top at its centre (hit-test), tag it, return its box."""
            return page.evaluate(
                """() => {
                  document.querySelectorAll('[data-evo-pick]').forEach(e => e.removeAttribute('data-evo-pick'));
                  const stage = document.getElementById('stage');
                  const sb = stage ? stage.getBoundingClientRect() : {left: 0, width: innerWidth};
                  const cats = Array.from(document.querySelectorAll('.cat')).map(el => {
                    const r = el.getBoundingClientRect();
                    return {el, x: r.left, y: r.top, w: r.width, h: r.height, cx: r.left + r.width / 2, cy: r.top + r.height / 2};
                  }).filter(c => c.w > 0 && c.h > 0);
                  const inner = cats.filter(c => c.cx > sb.left + 80 && c.cx < sb.left + sb.width - 80);
                  const order = (inner.length ? inner : cats).sort((a, b) => (b.y + b.h) - (a.y + a.h));
                  for (const c of order) {
                    const hit = document.elementFromPoint(c.cx, c.cy);
                    if (hit && (hit === c.el || c.el.contains(hit))) {
                      c.el.setAttribute('data-evo-pick', '1');
                      return {x: c.x, y: c.y, w: c.w, h: c.h};
                    }
                  }
                  if (order.length) { order[0].el.setAttribute('data-evo-pick', '1'); const c = order[0]; return {x: c.x, y: c.y, w: c.w, h: c.h}; }
                  return null;
                }"""
            )

        def picked_box():
            return page.evaluate(
                """() => { const el = document.querySelector('[data-evo-pick]'); if (!el) return null;
                           const r = el.getBoundingClientRect(); return {x: r.left, y: r.top, w: r.width, h: r.height}; }"""
            )

        def drag():
            b = pick_cat()
            if not b:
                return False, "no cats"
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
            sb = _stage_box(page) or {"x": 0, "w": 1000}
            sign = 1 if cx < sb["x"] + sb["w"] / 2 else -1  # drag toward the side with more room
            page.mouse.move(cx, cy)
            page.mouse.down()
            for i in range(1, 11):
                page.mouse.move(cx + sign * 20 * i, cy - 10 * i)
                page.wait_for_timeout(30)
            nb = picked_box() or b
            ncx, ncy = nb["x"] + nb["w"] / 2, nb["y"] + nb["h"] / 2
            page.mouse.up()
            ok = abs(ncx - (cx + sign * 200)) < 30 and abs(ncy - (cy - 100)) < 30
            return ok, f"pointer moved ({sign * 200:+d},-100); cat moved ({ncx - cx:+.0f},{ncy - cy:+.0f})"
        _guard(results, "drag_moves_cat", 2, drag)

        page.wait_for_timeout(2500)

        def throw(fast: bool):
            b = pick_cat()
            if not b:
                return -10**6
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
            page.mouse.move(cx, cy)
            page.mouse.down()
            steps, delay = (4, 10) if fast else (16, 60)
            for i in range(1, steps + 1):
                page.mouse.move(cx, cy - 160 * i / steps)
                page.wait_for_timeout(delay)
            page.mouse.up()
            apex = 10**9
            end = time.time() + 1.5
            while time.time() < end:
                pb = picked_box()
                if pb:
                    apex = min(apex, pb["y"])
                time.sleep(0.03)
            page.wait_for_timeout(2500)  # settle before the next throw
            return cy - 160 - apex  # extra height gained beyond the release point (px)

        def throw_scales():
            slow = throw(False)
            fast = throw(True)
            return fast > slow + 20, f"rise beyond release: slow={slow:.0f}px fast={fast:.0f}px"
        _guard(results, "throw_force_scales", 3, throw_scales)
    finally:
        ctx.close()

    return results
