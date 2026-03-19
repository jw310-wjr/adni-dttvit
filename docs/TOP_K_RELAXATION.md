# Approximate Differentiable Top-K (Proposal §3.4)

## Implementation

Our implementation uses **hard top-k selection + soft reweight** with straight-through:

1. **Forward**: Hard top-k selection (select k tokens by score).
2. **Backward**: Gumbel-Softmax weights on selected tokens; gradients flow to ScoreHead via straight-through.

This is an **approximate** differentiable Top-K relaxation, not a strictly fully differentiable soft top-k selection.

## Paper Wording

Recommend describing as:
- "approximate differentiable Top-K"
- "hard top-k with Gumbel-Softmax reweighting (straight-through)"

Avoid claiming "fully differentiable soft top-k" unless implementing a different formulation.
