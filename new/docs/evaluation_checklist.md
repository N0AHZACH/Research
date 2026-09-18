# Evaluation Checklist

Copy this checklist into the notes for any run that might support a manuscript claim.

## Identity

- [ ] Run ID:
- [ ] Git commit:
- [ ] Config path:
- [ ] Manifest path:
- [ ] Artifact directory:

## Matched Setup

- [ ] Same base model and revision across variants.
- [ ] Same tokenizer and revision across variants.
- [ ] Same training dataset, split, preprocessing, and sequence length.
- [ ] Same LoRA target modules and rank.
- [ ] Same optimizer, scheduler, batch size, gradient accumulation, and training-token budget.
- [ ] Same checkpoint selection rule.
- [ ] Same evaluation harness and task settings.

## Required Variants

- [ ] Pretrained base.
- [ ] Dense LoRA.
- [ ] Stochastic depth.
- [ ] Token DLR.
- [ ] Token DLR without KD, if this is a paper run.
- [ ] Random or parameter-free matched router, if this is a paper run.

## Metrics

- [ ] Task accuracy metrics written.
- [ ] Perplexity written with correct padding-label masking.
- [ ] Expected active layers written.
- [ ] Measured active layers written.
- [ ] Structural FLOP reduction written.
- [ ] Tokens/sec written.
- [ ] Peak GPU memory written.
- [ ] Router entropy written for DLR.
- [ ] Collapse flag written for DLR.

## Validity

- [ ] Routed models evaluated with routing active.
- [ ] Evaluator refuses silent full-depth fallback.
- [ ] Hook installation/removal verified.
- [ ] Raw task results saved.
- [ ] Summary CSV schema matches other variants.
- [ ] Environment and package versions saved.
- [ ] Run has at least the required seed count for its claim.

## Decision

- [ ] `verified`: usable for manuscript.
- [ ] `completed_unverified`: finished but not audited.
- [ ] `invalid`: do not use as evidence.
- [ ] `exploratory`: useful for debugging only.
