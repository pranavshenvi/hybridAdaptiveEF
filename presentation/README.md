# PercEF review deck

`PercEF_review.pptx` (15 slides, PES colours, speaker notes on every slide): problem statement ->
existing methodology -> our method -> novelty -> results -> limits and next steps.

Edit it directly in PowerPoint; every object is named in the Selection Pane. Placeholders to fill:
team names and guide on slide 1. Slide 13 (combined idea) still shows GloVe and MS Turing as
"running".

To regenerate from the script (overwrites manual edits):

    NODE_PATH=../poster/node_modules node build_deck.js
