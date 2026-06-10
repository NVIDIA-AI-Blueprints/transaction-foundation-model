# The Learning Path: Levels 100 → 400

This is the core curriculum. It explains the **same system four times**, each pass one level deeper — the way a good university sequence numbers its courses. The repetition is deliberate: concepts you accept on faith at Level 100 get mechanisms at Level 200, code at Level 300, and design rationale (including the skeletons in the closet) at Level 400.

| Level | Title | Question it answers | Time | Prerequisites |
|-------|-------|--------------------|------|---------------|
| [100](level-100-the-big-picture.md) | The Big Picture | *What is this and why should I care?* | 10 min | none |
| [200](level-200-the-building-blocks.md) | The Building Blocks | *What are the moving parts and how do they fit?* | 30 min | Level 100; primers [1](../02-concepts/01-foundation-models.md)–[3](../02-concepts/03-causal-language-modeling.md) as needed |
| [300](level-300-the-pipeline-in-code.md) | The Pipeline in Code | *Where exactly does each step happen, in which file?* | 60 min, ideally alongside the notebooks | Level 200; [environment set up](../01-getting-started/02-environment-setup.md) |
| [400](level-400-design-contracts-and-extensions.md) | Design Contracts & Extensions | *Why was it built this way, what are the sharp edges, and how do I change it safely?* | 45 min | Level 300 |

## How to use the levels

- **Don't skip 100 even if you're senior.** It establishes the vocabulary the other levels compress into.
- **Branch out to primers, then come back.** When a level leans on a concept (e.g., pooling), it links the [primer](../02-concepts/README.md). The levels stay narrative; the primers carry the conceptual depth.
- **Level 300 is best run, not just read.** Have notebooks 02–05 open; the section ordering matches them.
- **Level 400 is the price of admission for changing code.** If you're about to modify the tokenizer, config, or data pipeline — or train on new data ([Data section](../04-data/README.md)) — read it first. It documents the contracts that, when broken, fail *silently*.

After Level 400, you're equipped for the [Research](../05-research/README.md) and [Data](../04-data/README.md) sections, and for running your own experiments [with Loom](../06-experimentation/01-loom-workflow.md).
