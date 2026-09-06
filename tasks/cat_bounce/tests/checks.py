"""Playwright checks for the cat-bounce task.

Exposes run(browser, url, cfg) -> list[CheckResult]. Two fresh page loads:
  A: structure, physics (fall / decaying bounce / settle / bounds / no overlap), buttons, keyboard,
     pause, reset, localStorage persistence, resize recolour, errors, network.
  B: drag-follow and slow-vs-fast throw.
Every check is isolated with try/except so one failure cannot mask the others.
"""
from __future__ import annotations

from pathlib import Path

import re
import time
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    weight: float
    passed: bool
    detail: str = ""


SETTLE_S = 6.0  # spec says at rest within 5 s; we check at 5.5-6.1 s


# ------------------------------------------------------------------ helpers

def _boxes(page):
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('.cat')).map(el => {
             const r = el.getBoundingClientRect();
             return {x: r.left, y: r.top, w: r.width, h: r.height};
           })"""
    )


def _count(page):
    return page.evaluate("() => document.querySelectorAll('.cat').length")


def _stage_box(page):
    return page.evaluate(
        """() => { const s = document.getElementById('stage'); if (!s) return null;
                  const r = s.getBoundingClientRect();
                  return {x: r.left, y: r.top, w: r.width, h: r.height}; }"""
    )


def _counter_text(page):
    return page.evaluate("() => { const c = document.getElementById('counter'); return c ? c.textContent : null; }")


def _counter_ok(page):
    t = _counter_text(page)
    n = _count(page)
    return t is not None and t.strip() == f"Cats: {n}", f"counter={t!r} cats={n}"


def _sample_ys(page, seconds: float, step_s: float = 0.05):
    n = len(_boxes(page))
    traj = [[] for _ in range(n)]
    end = time.time() + seconds
    while time.time() < end:
        bs = _boxes(page)
        for i in range(min(n, len(bs))):
            traj[i].append(bs[i]["y"])
        time.sleep(step_s)
    return traj


def _max_motion(page, seconds: float):
    """Largest displacement of any cat (by index) over the window, in px."""
    bs0 = _boxes(page)
    best = 0.0
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(0.05)
        bs = _boxes(page)
        for a, b in zip(bs0, bs):
            best = max(best, abs(a["x"] - b["x"]), abs(a["y"] - b["y"]))
    return best


def _falls(ys):
    run = 0
    for a, b in zip(ys, ys[1:]):
        run = run + 1 if b > a + 3 else 0
        if run >= 3:
            return True
    return False


def _apexes(ys, min_rise=8.0):
    """Heights (as y, smaller = higher) of successive post-bounce apexes."""
    apexes = []
    i, n = 0, len(ys)
    while i < n - 1:
        # descend to a floor contact (local max of y)
        while i < n - 1 and ys[i + 1] >= ys[i] - 0.5:
            i += 1
        contact = ys[i]
        # rise to an apex (local min of y)
        j = i
        while j < n - 1 and ys[j + 1] <= ys[j] + 0.5:
            j += 1
        if j > i and contact - ys[j] >= min_rise:
            apexes.append(ys[j])
        if j == i:
            i += 1
        else:
            i = j
    return apexes


def _pick_cat(page):
    """Choose a cat that is actually on top at its centre (hit-test), tag it, return its box."""
    return page.evaluate(
        """() => {
          document.querySelectorAll('[data-evo-pick]').forEach(e => e.removeAttribute('data-evo-pick'));
          const stage = document.getElementById('stage');
          const sb = stage ? stage.getBoundingClientRect() : {left: 0, width: innerWidth};
          let pool = Array.from(document.querySelectorAll('.cat')).filter(el => !el.hasAttribute('data-evo-tried'));
          if (!pool.length) {
            document.querySelectorAll('[data-evo-tried]').forEach(e => e.removeAttribute('data-evo-tried'));
            pool = Array.from(document.querySelectorAll('.cat'));
          }
          const cats = pool.map(el => {
            const r = el.getBoundingClientRect();
            return {el, x: r.left, y: r.top, w: r.width, h: r.height, cx: r.left + r.width / 2, cy: r.top + r.height / 2};
          }).filter(c => c.w > 0 && c.h > 0);
          const inner = cats.filter(c => c.cx > sb.left + 80 && c.cx < sb.left + sb.width - 80);
          const order = (inner.length ? inner : cats).sort((a, b) => (b.y + b.h) - (a.y + a.h));
          for (const c of order) {
            const hit = document.elementFromPoint(c.cx, c.cy);
            if (hit && (hit === c.el || c.el.contains(hit))) {
              c.el.setAttribute('data-evo-pick', '1'); c.el.setAttribute('data-evo-tried', '1');
              return {x: c.x, y: c.y, w: c.w, h: c.h};
            }
          }
          if (order.length) { const c = order[0]; c.el.setAttribute('data-evo-pick', '1'); c.el.setAttribute('data-evo-tried', '1'); return {x: c.x, y: c.y, w: c.w, h: c.h}; }
          return null;
        }"""
    )


def _picked_box(page):
    return page.evaluate(
        """() => { const el = document.querySelector('[data-evo-pick]'); if (!el) return null;
                   const r = el.getBoundingClientRect(); return {x: r.left, y: r.top, w: r.width, h: r.height}; }"""
    )


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


# ------------------------------------------------------------------ main

def run_browser(browser, url, cfg) -> list[CheckResult]:
    results: list[CheckResult] = []
    errors: list[str] = []
    external: list[str] = []

    # ================================================================ page A
    ctx, page = _new_page(browser, url, cfg, errors, external)
    try:
        page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
        page.reload(wait_until="load")
        t_load = time.time()

        _guard(results, "title_mentions_cat", 1, lambda: ("cat" in page.title().lower(), page.title()[:60]))

        def stage_and_cats():
            sb = _stage_box(page)
            n = _count(page)
            return sb is not None and n >= 3, f"stage={'yes' if sb else 'no'} cats={n}"
        _guard(results, "stage_and_initial_cats", 2, stage_and_cats)

        def accessible():
            bad = page.evaluate(
                """() => Array.from(document.querySelectorAll('.cat'))
                        .filter(el => !(el.getAttribute('alt') || el.getAttribute('aria-label'))).length"""
            )
            n = _count(page)
            return n > 0 and bad == 0, f"{bad}/{n} cats missing alt/aria-label"
        _guard(results, "cats_accessible", 1, accessible)

        _guard(results, "counter_initial", 1, lambda: _counter_ok(page))

        # ---- physics sampling (first ~3 s)
        traj = _sample_ys(page, 3.0)
        fell_initial = sum(1 for ys in traj if _falls(ys))

        def cats_bounce_decay():
            n_bounce = sum(1 for ys in traj if len(_apexes(ys)) >= 1)
            decaying = 0
            for ys in traj:
                ap = _apexes(ys)
                if len(ap) >= 2 and ap[1] > ap[0] + 2:  # second apex is lower on screen = less height
                    decaying += 1
            return n_bounce >= 1 and decaying >= 1, f"{n_bounce}/{len(traj)} bounced, {decaying} with a lower 2nd apex"
        _guard(results, "cats_bounce_decay", 3, cats_bounce_decay)

        # ---- settle window
        remaining = SETTLE_S - 0.5 - (time.time() - t_load)
        if remaining > 0:
            page.wait_for_timeout(int(remaining * 1000))

        def cats_settle():
            moved_any = any(len(ys) > 2 and max(ys) - min(ys) > 5 for ys in traj)
            if not moved_any:
                return False, "cats never moved"
            m = _max_motion(page, 0.6)
            return m < 1.0, f"max displacement in [{SETTLE_S - 0.5:.1f}s,{SETTLE_S + 0.1:.1f}s] = {m:.1f}px"
        _guard(results, "cats_settle", 2, cats_settle)

        def in_bounds():
            sb = _stage_box(page)
            if not sb:
                return False, "no #stage"
            moved_any = any(len(ys) > 2 and max(ys) - min(ys) > 5 for ys in traj)
            if not moved_any:
                return False, "cats never moved, bounds check not earned"
            bs = _boxes(page)
            tol = 5
            out = [b for b in bs if b["x"] < sb["x"] - tol or b["y"] < sb["y"] - tol
                   or b["x"] + b["w"] > sb["x"] + sb["w"] + tol or b["y"] + b["h"] > sb["y"] + sb["h"] + tol]
            return len(bs) > 0 and not out, f"{len(out)}/{len(bs)} cats outside stage after settle"
        _guard(results, "cats_stay_in_bounds", 2, in_bounds)

        def no_overlap():
            bs = _boxes(page)
            worst = 0.0
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    a, b = bs[i], bs[j]
                    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
                    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
                    smaller = max(1.0, min(a["w"] * a["h"], b["w"] * b["h"]))
                    worst = max(worst, ix * iy / smaller)
            return len(bs) >= 2 and worst < 0.25, f"worst pairwise overlap {worst:.0%} of smaller cat"
        _guard(results, "cats_do_not_overlap", 2, no_overlap)

        # ---- rain button (also observe gravity on the new cats)
        rain_fell = [0]

        def rain():
            before = _count(page)
            btn = page.locator("#make-it-rain")
            if btn.count() == 0:
                return False, "no #make-it-rain"
            btn.first.click(timeout=3000)
            page.wait_for_timeout(150)
            rain_traj = _sample_ys(page, 0.6)[before:]
            rain_fell[0] = sum(1 for ys in rain_traj if _falls(ys))
            after = _count(page)
            return after - before >= 5, f"cats {before} -> {after}"
        _guard(results, "rain_adds_cats", 2, rain)

        def cats_fall():
            ok = fell_initial >= 1 or rain_fell[0] >= 1
            return ok, f"initial cats falling: {fell_initial}/{len(traj)}, rained cats falling: {rain_fell[0]}"
        _guard(results, "cats_fall", 2, cats_fall)

        _guard(results, "counter_after_rain", 1, lambda: _counter_ok(page))

        def space_rains():
            before = _count(page)
            page.mouse.click(5, page.viewport_size["height"] - 5)  # focus the page, away from buttons
            page.keyboard.press("Space")
            page.wait_for_timeout(400)
            after = _count(page)
            return after - before >= 5, f"cats {before} -> {after}"
        _guard(results, "space_key_rains", 1, space_rains)

        page.wait_for_timeout(2500)  # let the rained cats land

        def dblclick_removes():
            before = _count(page)
            b = _pick_cat(page)
            if not b:
                return False, "no cats"
            page.mouse.dblclick(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
            page.wait_for_timeout(300)
            after = _count(page)
            ok_counter, cdetail = _counter_ok(page)
            return after == before - 1 and ok_counter, f"cats {before} -> {after}; {cdetail}"
        _guard(results, "dblclick_removes_cat", 2, dblclick_removes)

        def pause_freezes():
            btn = page.locator("#pause")
            if btn.count() == 0:
                return False, "no #pause"
            label0 = btn.first.inner_text().strip()
            btn.first.click(timeout=3000)
            page.wait_for_timeout(100)
            label1 = btn.first.inner_text().strip()
            page.locator("#make-it-rain").first.click(timeout=3000)  # new cats must not fall while paused
            page.wait_for_timeout(150)
            frozen = _max_motion(page, 0.6)
            btn.first.click(timeout=3000)
            page.wait_for_timeout(100)
            label2 = btn.first.inner_text().strip()
            moving = _max_motion(page, 0.6)
            ok = ("resume" in label1.lower() and "pause" in label2.lower() and "pause" in label0.lower()
                  and frozen < 1.0 and moving > 5.0)
            return ok, f"labels {label0!r}->{label1!r}->{label2!r}; motion paused={frozen:.1f}px resumed={moving:.1f}px"
        _guard(results, "pause_freezes_stage", 2, pause_freezes)

        page.wait_for_timeout(2500)

        def reset_restores():
            btn = page.locator("#reset")
            if btn.count() == 0:
                return False, "no #reset"
            btn.first.click(timeout=3000)
            page.wait_for_timeout(300)
            n = _count(page)
            ok_counter, cdetail = _counter_ok(page)
            return n == 3 and ok_counter, f"cats after reset = {n}; {cdetail}"
        _guard(results, "reset_restores_three", 2, reset_restores)

        def persists():
            page.locator("#make-it-rain").first.click(timeout=3000)
            page.wait_for_timeout(300)
            before = _count(page)
            page.reload(wait_until="load")
            page.wait_for_timeout(500)
            after = _count(page)
            ok_counter, cdetail = _counter_ok(page)
            return before >= 8 and after == before and ok_counter, f"cats before reload {before}, after {after}; {cdetail}"
        _guard(results, "count_persists_reload", 2, persists)

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

        _guard(results, "loads_without_errors", 2, lambda: (not errors, "; ".join(errors)[:160]))
        _guard(results, "no_external_requests", 1, lambda: (not external, "; ".join(external)[:160]))
    finally:
        ctx.close()

    # ================================================================ page B: drag / throw
    ctx, page = _new_page(browser, url, cfg, [], [])
    try:
        page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
        page.reload(wait_until="load")
        page.wait_for_timeout(int(SETTLE_S * 1000))

        def drag_once():
            b = _pick_cat(page)
            if not b:
                return None
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
            sb = _stage_box(page) or {"x": 0, "w": 1000}
            sign = 1 if cx < sb["x"] + sb["w"] / 2 else -1
            page.mouse.move(cx, cy)
            page.mouse.down()
            for i in range(1, 11):
                page.mouse.move(cx + sign * 20 * i, cy - 10 * i)
                page.wait_for_timeout(30)
            nb = _picked_box(page) or b
            ncx, ncy = nb["x"] + nb["w"] / 2, nb["y"] + nb["h"] / 2
            page.mouse.up()
            ok = abs(ncx - (cx + sign * 200)) < 30 and abs(ncy - (cy - 100)) < 30
            moved = abs(ncx - cx) > 2 or abs(ncy - cy) > 2
            return ok, moved, f"pointer moved ({sign * 200:+d},-100); cat moved ({ncx - cx:+.0f},{ncy - cy:+.0f})"

        def drag():
            r = drag_once()
            if r is None:
                return False, "no cats"
            if not r[0] and not r[1]:  # grab failed outright: try one other cat
                page.wait_for_timeout(500)
                r2 = drag_once()
                if r2 is not None:
                    return r2[0], r2[2] + " (2nd cat)"
            return r[0], r[2]
        _guard(results, "drag_moves_cat", 2, drag)

        page.wait_for_timeout(3000)

        def throw(fast: bool):
            for attempt in range(2):
                b = _pick_cat(page)
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
                apex, moved = 10**9, False
                end = time.time() + 1.5
                while time.time() < end:
                    pb = _picked_box(page)
                    if pb:
                        apex = min(apex, pb["y"])
                        moved = moved or abs(pb["y"] - b["y"]) > 2 or abs(pb["x"] - b["x"]) > 2
                    time.sleep(0.03)
                page.wait_for_timeout(3000)
                if moved or attempt == 1:
                    return cy - 160 - apex  # px risen beyond the release point
            return -10**6

        def throw_scales():
            slow = throw(False)
            fast = throw(True)
            ok = slow > 5 and fast >= 2 * slow + 40
            return ok, f"rise beyond release: slow={slow:.0f}px fast={fast:.0f}px (need slow>5, fast>=2*slow+40)"
        _guard(results, "throw_force_scales", 3, throw_scales)
    finally:
        ctx.close()

    return results


# ---------------------------------------------------------------- harness contract (evolve.fitness)

def setup_worker(cfg):
    from playwright.sync_api import sync_playwright

    p = sync_playwright().start()
    browser = p.chromium.launch(headless=cfg.get("headless", True))
    return (p, browser)


def teardown_worker(ctx):
    p, browser = ctx
    try:
        browser.close()
    finally:
        p.stop()


def run(ctx, artifact_path, cfg):
    return run_browser(ctx[1], Path(artifact_path).resolve().as_uri(), cfg)
