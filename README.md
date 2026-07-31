# MoonDial: a bedtime storyteller with an LLM judge

A bedtime-story generator for ages 5-10, built on the required
`gpt-3.5-turbo`. The request is categorized and softened if needed, a
controlled twist makes the premise less predictable, the story is written
against a five-paragraph arc in an age-matched voice, and every draft must
survive two layers of judging: an LLM judge with per-criterion thresholds,
and plain-Python checks for the things a model cannot be trusted to count.
Failing drafts get targeted revision, bounded and measured.

```text
request 
        -
        
        
```

## How to run

```bash
pip install -r requirements.txt
```

Set your API key (never commit it):

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

Then:

```bash
python main.py                 
python main.py --read-aloud   
python eval.py                 
python -m unittest -v test_main  
```

`uv` users can replace `python` with `uv run python` in any command above.

## Block diagram

```
            ┌────────────────────────────────┐
            │ User request  (+ optional age) │
            └───────────────┬────────────────┘
                            ▼
            ┌────────────────────────────────┐
            │ Categorizer (LLM, t=0.2)       │
            │ category · age guess · softens │
            │ scary requests, never refuses  │
            └───────────────┬────────────────┘
                            ▼
            ┌────────────────────────────────┐
            │ Twist Engine (LLM, t=0.9)      │
            │ ONE gentle twist; role-guarded │
            │ (never swaps person/animal or  │
            │ who the story is about)        │
            └───────────────┬────────────────┘
                            ▼               35% chance
            ┌────────────────────────────────┐◄─────────┐
            │ Storyteller (LLM, t=0.9)       │   ┌──────┴───────┐
            │ 5-paragraph arc · named        │   │ Story memory │
            │ characters · age-matched voice │   │ (JSON file)  │
            │ · refrain or three-tries       │   └──────▲───────┘
            │ device · cliche ban            │          │ one element
            └───────────────┬────────────────┘          │ per story
                            ▼                           │
            ┌────────────────────────────────┐          │
            │ Length enforcement (code+LLM)  │          │
            │ code names each short paragraph│          │
            │ and its deficit; expander (≤2) │          │
            └───────────────┬────────────────┘          │
                            ▼                           │
              ≥2 structural problems? ──yes──►          │
              regenerate once, keep the                 │
              draft with fewer problems                 │
                            │                           │
                            ▼                           │
            ┌────────────────────────────────┐          │
            │ Judge (LLM, t=0) + Python      │          │
            │ sanity checks                  │          │
            │ 5 criteria with per-criterion  │          │
            │ thresholds (a low safety score │          │
            │ cannot be averaged away) plus  │          │
            │ word/paragraph floors and the  │          │
            │ quiet-ending check             │          │
            └───────────────┬────────────────┘          │
                            ▼                           │
              fixes? ──yes──► Reviser (≤2 rounds,       │
                 │            re-enforce length,        │
                 │            re-judge)                 │
                 no                                     │
                 ▼                                      │
            best draft by (passed, fewest               │
            fixes, judge score)                         │
                 ▼                                      │
            ┌────────────────────────────────┐          │
            │ Final story + score + passed   │          │
            └───────────────┬────────────────┘          │
                            ▼                           │
              "Anything you'd like changed?"            │
              feedback ──► revise ──► re-judge          │
                            │                           │
                            ▼                           │
            memory save (one background element) ───────┘
```

## Design decisions

- The judge returns rubric scores AND concrete fixes, and each criterion
  has its own threshold. A story that averages 9 but scores 5 on
  age-appropriateness fails - safety works as a gate rather than a
  weight. There is a unit test proving this.
- Drafts are ranked by `(passed, fewest outstanding fixes, judge score)`.
  Ranking by score alone let short, pretty drafts beat longer drafts that
  had actually fixed their problems - a bug found through the eval table
  and now pinned by a test.
- The word floor is enforced in code rather than requested in prose.
  Eval runs showed the model
  writing ~200-word stories regardless of how the length instruction was
  phrased, so code measures each paragraph and sends the expander a
  numbered list of deficits. Vague "make it longer" requests were a
  measured no-op.
- A structurally broken first draft (wrong paragraph count and far too
  short) triggers one regeneration instead of deep repair - a fresh
  sample at creative temperature usually beats two revisions of a bad
  skeleton.
- The stated age becomes concrete style rules (sentence length caps,
  sounds and repetition for younger listeners, light mystery for older
  ones). A bare age number in a prompt changes nothing.
- Creativity is engineered: the twist step names the overused patterns to
  avoid and asks for exactly one constrained change, with a guard so it
  can never swap who is a person and who is an animal.
- Thresholds (300 words minimum, 50 per paragraph) are calibrated to
  measured model output instead of aspiration. Earlier, stricter floors
  produced permanent failures the model could not reach.
- Every API call goes through one gateway with bounded retries; quota and
  auth errors fail fast instead of retrying pointlessly.

## Evaluation

`eval.py` runs six fixed requests - one per category plus a "terrifying
monster" request that must be softened instead of refused - and prints each
story's first-draft vs. final judge score, word count, pass/fail, and the
exact reason for any failure. Results from a real run:

```
request                                 category        score         words  passed  why not
----------------------------------------------------------------------------------------------------
A story about a submarine that explore  adventure       9.4 -> 9.4    304    True    -
A story about Alice and her best frien  animal_friend   9.0 -> 9.0    314    True    -
A silly story about a penguin who open  silly           9.6 -> 9.6    289    False   too_short
A quiet story about rain on a rooftop   calming         9.8 -> 9.8    317    True    -
A story that teaches my 8 year old abo  moral_lesson    8.8 -> 8.8    300    True    -
A terrifying story about a monster tha  adventure       9.2 -> 9.2    293    False   short_paragraph,too_short
```

Four of six pass every gate; the two failures miss the 300-word floor by
7-11 words, which the table reports honestly instead of hiding. The
remaining variance comes from sampling at creative temperature; the
production fix would be best-of-N generation, of which the structural
retry is the first step.

## Testing

`test_main.py` runs offline by replacing the model with scripted
responses. It verifies the machine rather than the model: the repair loop, the
per-criterion gates, the draft-ranking rule, the age override, the
expansion trigger, paragraph checks, and JSON extraction with braces
inside strings. `eval.py` answers the other question - whether the
prompts plus gpt-3.5 produce good stories - which is probabilistic and
costs money, so the two are kept separate.

## What the assignment asked for, and where it lives

| Requirement | Where |
| --- | --- |
| LLM judge improves the story | `judge_story` + revision loop in `make_story` |
| Story arcs | the storyteller prompt's 5-step arc |
| User feedback / request changes | `apply_feedback` + the loop in `main` |
| Categorize + tailored strategy | `CATEGORIES` + `categorize` |
| Ages 5-10 appropriate | softening in `categorize`, thresholds in `judge_passes`, `age_style` |
| Block diagram | above |
| Surprise features | twist engine, story memory, age-matched voice, read-aloud pacing, why-not diagnostics |
