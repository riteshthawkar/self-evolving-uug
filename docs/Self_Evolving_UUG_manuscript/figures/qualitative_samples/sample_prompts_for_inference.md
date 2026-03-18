# Qualitative Sample Prompts for Inference

This file lists the exact prompts/questions used for the supplementary qualitative panels so the runs can be reproduced directly.

Final shortlisted understanding candidates are also collected in:
`figures/qualitative_samples/understanding_top8/README.md`

For the current supplementary figures, the understanding and generation panels are shown in before/after format across the three backbones. The entries below list the exact questions and prompts used to create those panels.

## Understanding Panels

- `(a)` Source image: `open_images/000315.jpg`
  - Question: `What bib number is shown on the athlete in the back?`
  - Displayed after answer: `77`

- `(b)` Source image: `flickr30k/001191.jpg`
  - Question: `What vehicle is central in the image?`
  - Displayed after answer: `tricycle` / `cargo tricycle`

- `(c)` Source image: `nocaps/000588.jpg`
  - Question: `Which section is being actively filled?`
  - Displayed after answer: `Value Propositions`

## Additional Understanding Candidates

- `(e)` Source image: `open_images/000277.jpg`
  - Question: `What is immediately beside the black dishwasher?`
  - Base / majority: `counter` → `sink`

- `(f)` Source image: `open_images/000746.jpg`
  - Question: `What is the material of the label on the parking meter?`
  - Base / majority: `plastic` → `paper`

- `(g)` Source image: `nocaps/000527.jpg`
  - Question: `What is the material of the knife's handle?`
  - Base / majority: `plastic` → `rubber`

- `(h)` Source image: `flickr30k/001116.jpg`
  - Question: `What exact text is visible on the orange cone near the snowboarder?`
  - Base / majority: `r2` → `0`

- `(i)` Source image: `nocaps/000195.jpg`
  - Question: `What is the mug next to the lamp?`
  - Base / majority: `white mug` → `white mug with pens`

- `(j)` Source image: `nocaps/000326.jpg`
  - Question: `What is the liquid in the blender being blended into?`
  - Base / majority: `unknown` → `smoothie`

- `(k)` Source image: `flickr30k/000160.jpg`
  - Question: `What is the coconut lid?`
  - Base / majority: `straw holder` → `straw`

- `(l)` Source image: `open_images/000170.jpg`
  - Question: `What is the shadow cast by the tree branches on the stop sign?`
  - Base / majority: `dark shadow` → `partial shadow`

- `(m)` Source image: `nocaps/000231.jpg`
  - Question: `What is immediately beside the delorean's rear wheel?`
  - Base / majority: `do not enter sign` → `person`

- `(n)` Source image: `flickr30k/000801.jpg`
  - Question: `What is immediately beside the object under the blanket?`
  - Base / majority: `black shoes` → `black bag`

- `(o)` Source image: `flickr30k/000752.jpg`
  - Question: `What is immediately beside the diver's head?`
  - Base / majority: `oxygen tank` → `camera`

- `(p)` Source image: `flickr30k/000389.jpg`
  - Question: `What is immediately beside the person writing?`
  - Base / majority: `coffee cup` → `a cup and saucer`

## Generation Panels

- `(d)` Prompt: `A Lufthansa airplane on a tarmac, exactly 4 engines, blue tail`

- `(e)` Prompt: `A white duck with a red face swimming in water, with a black dog nearby`

- `(f)` Prompt: `Three people in a restaurant, with one woman pointing at a menu`
