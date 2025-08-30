---
annotations_creators:
  - expert-generated
language:
  - en
language_creators:
  - expert-generated
license: mit
multilinguality:
  - monolingual
pretty_name: Numina-Olympiads
size_categories:
  - 1K<n<10K
source_datasets:
  - AI-MO/NuminaMath-CoT
task_categories:
  - text-generation
  - mathematical-reasoning
task_ids:
  - math-word-problems
  - olympiad-math
paperswithcode_id: numina-olympiads
tags:
  - mathematics
  - olympiads
  - problem-solving
  - latex
  - mathematical-reasoning
  - math-word-problems
  - olympiad-math
metrics:
  - name: filtered_ratio
    type: ratio 
    value: 1.000
    description: Ratio of filtered dataset size to original dataset size
---

# Numina-Olympiads

Filtered NuminaMath-CoT dataset containing only olympiads problems with valid answers.

## Dataset Information
- Split: train
- Original size: 32926
- Filtered size: 32926
- Source: olympiads
- All examples contain valid boxed answers

## Dataset Description
This dataset is a filtered version of the NuminaMath-CoT dataset, containing only problems from olympiad sources that have valid boxed answers. Each example includes:
- A mathematical word problem
- A detailed solution with step-by-step reasoning
- A boxed final answer in LaTeX format

## Usage
The dataset is particularly useful for:
- Training and evaluating math problem-solving models
- Studying olympiad-style mathematical reasoning
- Testing model capabilities on complex word problems
