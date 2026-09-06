Build a single-page web toy inspired by "cat bounce".

Deliver exactly one self-contained file, index.html, with all CSS and JavaScript
inline. The page must work offline when opened from disk: no external scripts,
stylesheets, fonts, images or network requests of any kind. Draw the cats with
inline SVG, emoji, or CSS.

Behaviour:
- The page title contains the word "Cat".
- A full-viewport container with id="stage" holds the cats.
- Each cat is an element with class="cat", absolutely positioned inside #stage.
  At least 3 cats are present when the page loads. Each cat has an alt or
  aria-label attribute.
- Cats are affected by gravity: they fall, bounce off the bottom of the stage
  with some energy loss, and eventually come to rest. Cats bounce off the left
  and right edges too and never leave the stage.
- Cats can be picked up with the mouse (mousedown on a cat, move, mouseup).
  While held, the cat follows the pointer. When released, the cat keeps the
  pointer's velocity, so a faster throw sends it higher and further.
- A button with id="make-it-rain" adds at least 5 new cats falling from the top
  of the stage each time it is clicked.
- The background colour of the page changes every time the window is resized.

Respond with the complete index.html inside a single ```html code block and
nothing else.
