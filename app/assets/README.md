# Interface artwork

## `motif.svg` — the background motif

Original artwork. Nested open arcs enclosing a small circle, with a sprig
above: something held, and something growing.

### Why not a photograph

Deliberately **non-figurative**, for three reasons in order of weight.

**1. Consent.** A photograph of a pregnant woman in a risk-screening tool means
showing a real, identifiable person next to the words "high risk". No stock
licence covers appearing in what looks like a clinical triage screen, and no
model release anticipates it. The safest handling of a real person's likeness
here is not to use one.

**2. Abstraction travels; depiction excludes.** A drawing of a body invites the
viewer to check whether it looks like her — her age, her dress, her region — and
any mismatch quietly signals "this is not for you". For a tool whose whole
purpose is that a woman in rural Punjab acts on what it says, that is a real
cost. Arcs carry care without claiming to depict anyone.

**3. Local appropriateness.** Non-figurative imagery avoids sensitivities around
depicting the human form that are genuine for part of the intended audience in
Pakistan. An abstract motif is legible to everyone and objectionable to nobody.

### Rendering constraints

* **Inlined as a base64 data URI**, not served over HTTP, so the interface looks
  identical offline. The deployment setting this is written for cannot assume a
  network connection.
* **Held at 14% opacity** on its own compositing layer, separate from the colour
  wash, so the artwork can stay faint without also fading the gradients.
* **Never behind body text.** Message cards, criteria cards, and notices are all
  opaque. In a screening tool legibility is a safety property, not an aesthetic
  preference — a health worker misreading a referral timeframe because it sat on
  a busy background is a clinical failure, not a design flaw.
* **The file must begin with `<svg>`.** An SVG that opens with an XML comment is
  valid XML but fails to decode when used as a CSS background image — which is
  exactly how the first version of this file broke, silently and invisibly.
  Documentation lives in this README for that reason; the SVG carries `<title>`
  and `<desc>` instead, which are legal inside the root element and also serve
  assistive technology.

### Colour

Single accent, `#1f6f8b`, matching the interface primary. The motif never uses a
risk-band colour — green, amber, and red mean exactly one thing on this screen,
and decoration must not borrow them.
