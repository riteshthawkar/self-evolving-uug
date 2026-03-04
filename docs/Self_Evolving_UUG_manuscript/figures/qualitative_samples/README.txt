Qualitative Figure Reference — Self-Evolving UUG
=================================================

PANEL A: Understanding — Self-Consistency Correction
----------------------------------------------------
Shows how the SC mechanism (N=7 prompt perturbations) corrects
single greedy-pass errors. Data from E1_main_joint/iter_log.jsonl.

(a) Object Identification
    Image: understanding/a_object_identification__bed_to_table.jpg
    Source: vsr/000132.jpg  |  Step 588
    Question: "What is the suitcase lid resting on?"
    Greedy answer:  bed          (WRONG)
    SC answer:      table        (6/7 agree)

(b) Spatial Reasoning
    Image: understanding/b_spatial_reasoning__tree_trunk_to_rock.jpg
    Source: realworldqa/000405.jpg  |  Step 597
    Question: "What is immediately beside the signpost near the trail?"
    Greedy answer:  tree trunk   (WRONG)
    SC answer:      rock         (6/7 agree)

(c) Action Recognition
    Image: understanding/c_action_recognition__preparing_food_to_slicing_meat.jpg
    Source: open_images/000063.jpg  |  Step 138
    Question: "What is the man bending over the cutting board doing?"
    Greedy answer:  preparing food   (VAGUE)
    SC answer:      slicing meat     (5/7 agree)

(d) Hallucination Correction
    Image: understanding/d_hallucination__transport_nutrients_to_tree_bark.jpg
    Source: flickr30k/000961.jpg  |  Step 117
    Question: "What is the function of the part of the tree trunk the child is touching?"
    Greedy answer:  transport of nutrients   (HALLUCINATED)
    SC answer:      tree bark                (5/7 agree)


PANEL B: Generation — QA Fidelity Assessment
---------------------------------------------
Shows how the Solver detects generation failures via diagnostic QA.
Data from BAGEL_exp/generation_rollouts.jsonl.

(e) Counting + Text
    Source: generation_source/e_counting_text__lufthansa.jpg    (nocaps/000008.jpg)
    Generated: generation_generated/e_counting_text__lufthansa.png  (step_001884_000008_cand0.png)
    Prompt: "Lufthansa airplane on tarmac, four engines, blue tail"
    QA Results:
      FAIL  "What airline?"       expected: lufthansa  |  got: hdasthan
      FAIL  "How many engines?"   expected: four       |  got: two

(f) Counting + Semantic
    Source: generation_source/f_counting_semantic__restaurant.jpg    (flickr30k/001864.jpg)
    Generated: generation_generated/f_counting_semantic__restaurant.png  (step_001865_001864_cand0.png)
    Prompt: "Three people in a restaurant, one woman pointing at a menu"
    QA Results:
      FAIL  "How many people?"        expected: three  |  got: two
      FAIL  "What is she pointing at?" expected: menu   |  got: chandelier
      PASS  "Chair color?"            expected: red    |  got: red

(g) Color Attribution
    Source: generation_source/g_color_attribution__duck_dog.jpg    (flickr30k/000188.jpg)
    Generated: generation_generated/g_color_attribution__duck_dog.png  (step_000189_000188_cand0.png)
    Prompt: "White duck with a red face swimming; black dog nearby"
    QA Results:
      FAIL  "Duck's face color?"  expected: red    |  got: white
      PASS  "Dog color?"          expected: black  |  got: black

(h) Object Count
    Source: generation_source/h_object_count__three_dogs.jpg    (flickr30k/000008.jpg)
    Generated: generation_generated/h_object_count__three_dogs.png  (step_000009_000008_cand0.png)
    Prompt: "Three dogs playing with a ball in an outdoor area"
    QA Results:
      FAIL  "How many dogs?"    expected: three           |  got: five
      PASS  "Dogs' activity?"   expected: playing w/ ball |  got: playing w/ ball
