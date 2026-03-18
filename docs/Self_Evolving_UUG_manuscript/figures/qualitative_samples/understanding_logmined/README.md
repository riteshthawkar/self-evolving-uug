# Log-mined supplementary understanding candidates

These candidates were mined from the understanding logs by selecting cases where the base/intuitive answer differs from the majority self-consistent answer with reasonably strong agreement. They are intended as additional supplementary-only understanding samples.

## Candidates

- `(e)` `e_relation__dishwasher_sink.jpg`
  - Source image: `joint_pool_10k/images/open_images/000277.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `1182`
  - Question: `What is immediately beside the black dishwasher?`
  - Base/intuitive answer: `counter`
  - Majority answer: `sink`
  - Majority fraction: `5/7`
  - Why it is useful: local spatial relation in a cluttered kitchen scene

- `(f)` `f_ocr_material__parking_meter_label.jpg`
  - Source image: `joint_pool_10k/images/open_images/000746.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `859`
  - Question: `What is the material of the label on the parking meter?`
  - Base/intuitive answer: `plastic`
  - Majority answer: `paper`
  - Majority fraction: `5/7`
  - Why it is useful: local material recognition around text-heavy signage

- `(g)` `g_local_attribute__knife_handle.jpg`
  - Source image: `joint_pool_10k/images/nocaps/000527.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `893`
  - Question: `What is the material of the knife's handle?`
  - Base/intuitive answer: `plastic`
  - Majority answer: `rubber`
  - Majority fraction: `5/7`
  - Why it is useful: fine-grained local attribute recognition on a simple crop

- `(h)` `h_ocr_local__orange_cone_text.jpg`
  - Source image: `joint_pool_10k/images/flickr30k/001116.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `1384`
  - Question: `What exact text is visible on the orange cone near the snowboarder?`
  - Base/intuitive answer: `r2`
  - Majority answer: `0`
  - Majority fraction: `6/7`
  - Why it is useful: small-text OCR on a non-dominant object

- `(i)` `i_local_state__mug_with_pens.jpg`
  - Source image: `joint_pool_10k/images/nocaps/000195.jpg`
  - Log source: `runs/final/E1_main_joint/iter_log.jsonl`, step `398`
  - Question: `What is the mug next to the lamp?`
  - Base/intuitive answer: `white mug`
  - Majority answer: `white mug with pens`
  - Majority fraction: `5/7`
  - Why it is useful: local object-state recognition; the base answer misses the contents of the mug

- `(j)` `j_result_state__blender_smoothie.jpg`
  - Source image: `joint_pool_10k/images/nocaps/000326.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `1028`
  - Question: `What is the liquid in the blender being blended into?`
  - Base/intuitive answer: `unknown`
  - Majority answer: `smoothie`
  - Majority fraction: `5/7`
  - Why it is useful: result-state inference from a close-up kitchen scene

- `(k)` `k_local_object__coconut_straw.jpg`
  - Source image: `joint_pool_10k/images/flickr30k/000160.jpg`
  - Log source: `runs/final/E1_main_joint/iter_log.jsonl`, step `601`
  - Question: `What is the coconut lid?`
  - Base/intuitive answer: `straw holder`
  - Majority answer: `straw`
  - Majority fraction: `5/7`
  - Why it is useful: small local object identification in a busy restaurant scene

- `(l)` `l_local_shadow__stop_sign.jpg`
  - Source image: `joint_pool_10k/images/open_images/000170.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `883`
  - Question: `What is the shadow cast by the tree branches on the stop sign?`
  - Base/intuitive answer: `dark shadow`
  - Majority answer: `partial shadow`
  - Majority fraction: `5/7`
  - Why it is useful: subtle local attribute reasoning under partial occlusion

## Notes

- These are good candidates for a `before / after` supplementary figure because the base answer is clearly worse than the self-consistent/evolved answer.
- If you want a slightly cleaner wording for panel `(h)`, you can also use: `What number is visible on the orange cone near the snowboarder?`
- The strongest added cases from this second batch are `(i)`, `(j)`, and `(k)`. `(l)` is usable, but visually subtler than the others.
