# Collaborative Story Generator (GPT-J)

Interactive multi-chapter story generation system built with GPT-J-6B
and HuggingFace Transformers.

## Overview

This project implements a human-AI collaborative story generation system
using the GPT-J-6B language model. Users select story background,
emotional tone, number of chapters, and optionally contribute text
before each chapter. The model dynamically generates coherent narrative
content while preserving thematic consistency.

Key features: - Background selection or custom story setting -
Emotion-controlled chapter generation - Multi-chapter story creation -
Human-in-the-loop collaborative writing - Automatic story ending
generation - Context truncation for long narratives

## Model

-   Model: EleutherAI/gpt-j-6B
-   Framework: HuggingFace Transformers
-   Backend: PyTorch

The system generates text using temperature sampling and nucleus
sampling (top-p), with repetition penalty to reduce repetitive outputs.

## Context Management

GPT-J has a limited context window. To maintain coherence while
controlling memory usage, the system applies dynamic context truncation
(max 1500 tokens), retaining recent and relevant content.

## Content Control

To prevent non-narrative outputs (e.g., Q&A patterns), the system: -
Uses generation parameter tuning (temperature, top-p, repetition
penalty) - Applies simple pattern filtering and regeneration when needed

## System Modules

1.  User Interaction Module
    -   Background selection (preset or custom)
    -   Emotion selection per chapter
    -   Chapter length control
    -   User-written story fragments
2.  Story Generation Module
    -   GPT-J-based text generation
    -   Multi-chapter narrative continuation
3.  Context Management Module
    -   Dynamic truncation
    -   Coherence preservation
4.  Output Module
    -   Automatic ending generation
    -   Saves final story as generated_story.txt

## Hardware Environment

CPU: Intel Core i9-13900HX\
GPU: NVIDIA RTX 4090 Laptop\
RAM: 32GB

## Software Environment

OS: Windows 11\
Python: 3.9+

Libraries: - transformers - torch - sklearn

## Repository Structure

code.py \# Main collaborative story generation system
generated_story.txt \# Output file (generated at runtime)

## How to Run

1.  Install dependencies:

pip install transformers torch scikit-learn

2.  Run the program:

python code.py

3.  Follow the terminal prompts to:
    -   Choose background
    -   Select emotion
    -   Define story length
    -   Optionally add user text

The system will generate and save the complete story.

## Limitations

-   GPT-J context window limitations
-   Quality degradation for very long stories
-   High GPU memory requirement
-   Potential thematic drift in long narratives

## Future Improvements

-   Improved long-context handling
-   Multi-character dialogue modeling
-   Reinforcement learning with user feedback
-   Branching narrative structures
-   Multimodal extensions (image/audio input)

## Author

Dongjie Chen\
MSc Computer Science (Data Science)\
Leiden University
