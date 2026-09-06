Build a single-page web toy inspired by "cat bounce".

Deliver exactly one self-contained file, index.html, with all CSS and JavaScript
inline. The page must work offline when opened from disk: no external scripts,
stylesheets, fonts, images or network requests of any kind. Draw the cats with
inline SVG, emoji, or CSS.

Structure:
- The page title contains the word "Cat".
- A full-viewport container with id="stage" holds the cats.
- Each cat is an element with class="cat", absolutely positioned inside #stage,
  with an alt or aria-label attribute. At least 3 cats are present on load.
- An element with id="counter" whose text is exactly "Cats: N", where N is the
  current number of cats. It must update immediately whenever cats are added or
  removed.
- A button with id="make-it-rain", a button with id="pause" and a button with
  id="reset".

Physics:
- Cats are affected by gravity: they fall and bounce off the bottom of the
  stage. Every bounce loses energy, so each successive bounce is lower than
  the last, and with no interaction every cat is completely at rest within 5
  seconds of the page loading.
- Cats bounce off the left and right edges and never leave the stage.
- Cats collide with each other: two cats never overlap while at rest.

Interaction:
- Cats can be picked up with the mouse (mousedown on a cat, move, mouseup).
  While held, the cat follows the pointer exactly. When released, the cat keeps
  the pointer's velocity, so a faster throw sends it noticeably higher and
  further than a slow one.
- Double-clicking a cat removes it.
- Clicking #make-it-rain, or pressing the Space key, adds at least 5 new cats
  falling from the top of the stage.
- #pause toggles the simulation. Its label reads "Pause" while running and
  "Resume" while paused. While paused nothing on the stage moves, including
  newly added cats.
- #reset removes all cats, adds exactly 3 fresh ones, and clears saved state.
- The number of cats is saved in localStorage and restored on reload, so a
  reload shows the same number of cats as before.
- The background colour of the page changes every time the window is resized.

Respond with the complete index.html inside a single ```html code block and
nothing else.
