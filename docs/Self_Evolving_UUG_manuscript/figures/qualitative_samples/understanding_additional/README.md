# Additional understanding candidates for supplementary qualitative figures

These are additional log-mined understanding samples chosen to match the current `before / after` supplementary figure style: local relation, local object-state, OCR-adjacent detail, and fine-grained non-dominant evidence. All are sourced from the training image pool and were selected from cases where the intuitive/base answer differs from the self-consistent majority answer.

- `(a)` `a_mug_with_pens.jpg`
  - Source image: `joint_pool_10k/images/nocaps/000195.jpg`
  - Log source: `runs/final/E1_main_joint/iter_log.jsonl`, step `398`
  - Question: `What is the mug next to the lamp?`
  - Base/intuitive answer: `white mug`
  - Majority answer: `white mug with pens`
  - Why it is useful: local object-state; the correction comes from noticing mug contents.

- `(b)` `b_blender_smoothie.jpg`
  - Source image: `joint_pool_10k/images/nocaps/000326.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `1028`
  - Question: `What is the liquid in the blender being blended into?`
  - Base/intuitive answer: `unknown`
  - Majority answer: `smoothie`
  - Why it is useful: result-state inference from a close-up kitchen scene.

- `(c)` `c_coconut_straw.jpg`
  - Source image: `joint_pool_10k/images/flickr30k/000160.jpg`
  - Log source: `runs/final/E1_main_joint/iter_log.jsonl`, step `601`
  - Question: `What is the coconut lid?`
  - Base/intuitive answer: `straw holder`
  - Majority answer: `straw`
  - Why it is useful: small local object identification in a cluttered dining scene.

- `(d)` `d_stop_sign_shadow.jpg`
  - Source image: `joint_pool_10k/images/open_images/000170.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `883`
  - Question: `What is the shadow cast by the tree branches on the stop sign?`
  - Base/intuitive answer: `dark shadow`
  - Majority answer: `partial shadow`
  - Why it is useful: subtle local attribute reasoning under partial occlusion.

- `(e)` `e_delorean_rear_wheel_person.jpg`
  - Source image: `joint_pool_10k/images/nocaps/000231.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `93`
  - Question: `What is immediately beside the delorean's rear wheel?`
  - Base/intuitive answer: `do not enter sign`
  - Majority answer: `person`
  - Why it is useful: local spatial relation in a visually busy scene.

- `(f)` `f_blanket_black_bag.jpg`
  - Source image: `joint_pool_10k/images/flickr30k/000801.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `850`
  - Question: `What is immediately beside the object under the blanket?`
  - Base/intuitive answer: `black shoes`
  - Majority answer: `black bag`
  - Why it is useful: non-dominant object grounding in a crowded outdoor scene.

- `(g)` `g_diver_head_camera.jpg`
  - Source image: `joint_pool_10k/images/flickr30k/000752.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `927`
  - Question: `What is immediately beside the diver's head?`
  - Base/intuitive answer: `oxygen tank`
  - Majority answer: `camera`
  - Why it is useful: local relation around overlapping equipment.

- `(h)` `h_person_writing_cup_saucer.jpg`
  - Source image: `joint_pool_10k/images/flickr30k/000389.jpg`
  - Log source: `runs/final/E2_understanding_only/iter_log.jsonl`, step `931`
  - Question: `What is immediately beside the person writing?`
  - Base/intuitive answer: `coffee cup`
  - Majority answer: `a cup and saucer`
  - Why it is useful: fine local object recognition in an indoor tabletop scene.
