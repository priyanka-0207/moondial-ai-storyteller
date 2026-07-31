import argparse
import json
import os
import random
import time

"""
MoonDial - The Bedtime storyteller for children between the ages 5-10 with an LLM judge.

Flow: categorize -> one controlled twist -> generate in an age-matched
voice -> LLM judge with per-criterion thresholds + Python sanity checks
-> bounded revision -> best draft by (passed, fewest fixes, score).

if I had spent 2 more hours on this project, I will build the following:

1. Best-of-N generation to cut run-to-run variance: sample two or three
   drafts up front and judge the best one, instead of repairing a single
   sample. The structural-retry fallback already implements the first
   step of this.
2. Grow the eval set beyond six requests and append every run's scores to
   a history file, so the gate thresholds keep being tuned on data. The
   current floors (300 words, 50 per paragraph) were calibrated this way.
3. A deterministic character check: scan dialogue attributions against
   the names in the request, so a dropped or renamed character is caught
   by code instead of relying on the judge's request_fidelity score.
"""

MODEL = "gpt-3.5-turbo"
MAX_REVISIONS = 2
EXACT_PARAGRAPHS = 5
MIN_PARAGRAPH_WORDS = 50
MIN_WORDS = 300
MAX_WORDS = 650

def age_style(age):
    if age <= 6:
        return ("very simple words a 5-year-old knows, sentences under 8 "
                "words, playful sounds (swish, ting, plop), and a little "
                "repetition a small child can anticipate")
    if age <= 8:
        return ("simple words, sentences under 12 words, playful dialogue, "
                "and light humor")
    return ("clear words with an occasional rich one, sentences under 14 "
            "words, cleverer jokes, and a touch of mystery")

JUDGE_THRESHOLDS = {
    "request_fidelity": 8,
    "age_appropriate": 9,
    "story_arc": 8,
    "calm_ending": 8,
    "engaging": 7,
}

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "story_memory.json")
MEMORY_REUSE_CHANCE = 0.35

CATEGORIES = {
    "adventure": "a fun quest with clever, never dangerous challenges",
    "animal_friend": "a warm animal friendship with small acts of kindness",
    "silly": "playful surprises, gentle jokes, and absurd moments",
    "calming": "a slow, soothing story with soft images and quiet sounds",
    "moral_lesson": "show the lesson through actions, never a moral speech",
}

# LLM
def call_model(prompt, temperature=0.8, max_tokens=1600):
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    last_error = None
    for attempt in range(3):
        try:
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message["content"]

        except Exception as error:
            message = str(error).lower()
            if any(code in message for code in (
                "insufficient_quota", "invalid_api_key", "model_not_found"
            )):
                raise RuntimeError(f"LLM call failed: {error}") from error
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")

def extract_json(text):
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:index + 1])
    raise ValueError("unbalanced JSON object")

def call_json(prompt, temperature=0.2):
    text = call_model(prompt, temperature)
    try:
        return extract_json(text)
    except (ValueError, json.JSONDecodeError):
        corrected = call_model(
            "Reply with ONLY the corrected JSON object.\n\n" + text,
            temperature=0.0,
        )
        return extract_json(corrected)

# components
def categorize(request):
    return call_json(f"""You prepare bedtime-story requests for ages 5-10.

Request: {request}
If it is scary, violent, or inappropriate, soften it into the closest cozy,
child-friendly request while preserving harmless names and themes.

Return ONLY JSON:
{{"category": "<adventure, animal_friend, silly, calming, or moral_lesson>",
  "age": <5-10, default 7>,
  "safe_request": "<softened only if necessary>",
  "was_softened": <true or false>}}""", temperature=0.2)

def add_twist(safe_request):
    result = call_json(f"""Make this bedtime-story premise more original.

Request: {safe_request}

Preserve all names, characters, relationships, and the topic. Never change
who is a person and who is an animal, never swap traits between characters,
and never change who the story is about. Avoid generic magic kingdoms, wise
elders who solve everything, and "it was all a dream."
Add exactly ONE gentle twist: give magic a funny limitation, make an
apparent villain helpful, or let the hero's weakness turn out useful.
Nothing scary.

Return ONLY JSON: {{"premise": "<one sentence>"}}""", temperature=0.9)
    return result.get("premise") or safe_request

def recall_memory():
    try:
        with open(MEMORY_FILE, encoding="utf-8") as file:
            items = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if items and random.random() < MEMORY_REUSE_CHANCE:
        return random.choice(items)
    return None

def save_memory(story):
    element = call_json(f"""Pick ONE small harmless background element from
this story: a place, object, or side character. Never choose the hero, main
conflict, or an object that solves the conflict.

Story:
{story}

Return ONLY JSON:
{{"name": "<short name>", "description": "<one sentence>"}}""")
    try:
        with open(MEMORY_FILE, encoding="utf-8") as file:
            items = json.load(file)
    except (OSError, json.JSONDecodeError):
        items = []

    name = element.get("name")
    existing = {str(item.get("name", "")).casefold() for item in items}
    if name and name.casefold() not in existing:
        items.append(element)
        with open(MEMORY_FILE, "w", encoding="utf-8") as file:
            json.dump(items[-10:], file, indent=2)

def tell_story(premise, category, age, memory=None):
    strategy = CATEGORIES.get(category, CATEGORIES["animal_friend"])
    memory_line = ""
    if memory:
        memory_line = (
            f"\nOptional background detail: {memory['name']} - "
            f"{memory['description']}. Never make it the main plot "
            f"or solution."
        )
    header = f"You are a warm bedtime storyteller writing for age {age}."
    return call_model(f"""{header}

Premise: {premise}
Category style: {strategy}.{memory_line}
Voice for this age: {age_style(age)}.

Write exactly {EXACT_PARAGRAPHS} paragraphs and {MIN_WORDS}-{MAX_WORDS} words
total. Each paragraph must have at least {MIN_PARAGRAPH_WORDS} words (aim
for 90).

1. Introduce named characters and one small problem. Every character gets a
   name (a princess is "Princess Maya", never just "the princess"); use any
   requested names exactly.
2. Increase curiosity while they start solving it.
3. Reach the most active moment, but include no real danger.
4. Resolve every important setup and begin slowing down.
5. Become quiet and sleepy: short sentences, warm images, no exclamation
   marks, and the hero ends safe and drowsy.

Use ONE classic tale device: either a short line that repeats like a refrain
the child can say along, or three tries where the third one finally works.
The premise's twist must be visible in what actually happens.

NEVER use: "Once upon a time", a generic magical kingdom, an evil witch or
sorceress, a chosen one, or "happily ever after".

Preserve every requested name, relationship, topic, setting, and lesson. Show
lessons through actions, not speeches. Return only the story.""",
                      temperature=0.9, max_tokens=1800).strip()

def judge_story(story, age, original_request, premise):
    return call_json(f"""You are a children's librarian evaluating a bedtime
story for a {age}-year-old.

Approved safe request: {original_request}
Planned premise: {premise}

Story:
{story}

Be strict but evidence-based. Do not invent problems. A strong story may have
no fixes. Score 1-10:
- request_fidelity: requested names, relationships, topic, setting, and
  lesson. If a requested name is missing, or a person/animal role or a
  relationship was changed, this score is 4 or lower.
- age_appropriate: safe, kind, reassuring, suitable vocabulary
- story_arc: clear setup, development, active moment, and resolution
- calm_ending: final paragraph becomes quiet and sleepy
- engaging: specific details, enough curiosity for a child, and every
  character has a name a child can hold onto (never "the princess")

Return ONLY JSON:
{{"scores": {{"request_fidelity": <1-10>, "age_appropriate": <1-10>,
  "story_arc": <1-10>, "calm_ending": <1-10>, "engaging": <1-10>}},
  "fixes": ["<specific minimal repair>", ...]}}""", temperature=0.0)

def revise_story(story, fixes, age):
    fix_list = "\n".join(f"- {fix}" for fix in fixes)
    return call_model(f"""Revise this bedtime story for age {age}.

Apply ONLY these fixes:
{fix_list}

Preserve all unaffected names, relationships, plot events, and tone. Keep
exactly {EXACT_PARAGRAPHS} paragraphs, at least {MIN_PARAGRAPH_WORDS} words per
paragraph, and {MIN_WORDS}-{MAX_WORDS} words total. The final paragraph must
stay quiet and contain no exclamation marks.

Story:
{story}

Return only the revised story.""", temperature=0.5, max_tokens=1800).strip()

def paragraphs(story):
    return [part.strip() for part in story.strip().split("\n\n")
            if part.strip()]

def length_problems(story):
    problems = []
    parts = paragraphs(story)
    if len(parts) != EXACT_PARAGRAPHS:
        problems.append(f"Restructure into exactly {EXACT_PARAGRAPHS} "
                        f"paragraphs (currently {len(parts)}).")
    for index, part in enumerate(parts, 1):
        count = len(part.split())
        if count < MIN_PARAGRAPH_WORDS:
            problems.append(
                f"Paragraph {index} has {count} words; grow it to at least "
                f"{MIN_PARAGRAPH_WORDS + 20} words with sensory details and "
                f"brief dialogue."
            )
    if len(story.split()) < MIN_WORDS:
        problems.append(f"The whole story has {len(story.split())} words; "
                        f"it must reach at least {MIN_WORDS}.")
    return problems

def expand_story(story, age, problems):
    target_list = "\n".join(f"- {problem}" for problem in problems)
    return call_model(f"""Expand this bedtime story for age {age} without
changing its names, relationships, or plot events. Voice: {age_style(age)}.

Fix exactly these length problems:
{target_list}

Add only sensory details, small reactions, and brief dialogue inside the
existing paragraphs. Add no new characters, events, or magic. Keep the final
paragraph quiet and free of exclamation marks.

Story:
{story}
Return only the expanded story.""", temperature=0.6, max_tokens=1800).strip()


def ensure_length(story, age, verbose=False):
    for _ in range(2):
        problems = length_problems(story)
        if not problems:
            break
        story = expand_story(story, age, problems)
        if verbose:
            print(f"[expanded to {len(story.split())} words]")
    return story


def average(scores):
    values = [float(value) for value in scores.values()
              if isinstance(value, (int, float))]
    return round(sum(values) / max(1, len(values)), 1)


def sanity_checks(story):
    problems = list(length_problems(story))
    parts = paragraphs(story)
    words = len(story.split())

    if words > MAX_WORDS:
        problems.append(
            f"Story is {words} words; trim it to under {MAX_WORDS}."
        )
    if parts and "!" in parts[-1]:
        problems.append("Remove exclamation marks from the final paragraph.")

    return problems

def failure_reasons(result):
    reasons = []
    scores = result["verdict"].get("scores", {})
    for name, threshold in JUDGE_THRESHOLDS.items():
        try:
            if float(scores.get(name, 0)) < threshold:
                reasons.append(name)
        except (TypeError, ValueError):
            reasons.append(f"{name}?")
    story = result["story"]
    parts = paragraphs(story)
    if len(parts) != EXACT_PARAGRAPHS:
        reasons.append("paragraph_count")
    if any(len(part.split()) < MIN_PARAGRAPH_WORDS for part in parts):
        reasons.append("short_paragraph")
    if len(story.split()) < MIN_WORDS:
        reasons.append("too_short")
    if len(story.split()) > MAX_WORDS:
        reasons.append("too_long")
    if parts and "!" in parts[-1]:
        reasons.append("loud_ending")
    judge_fixes = [fix for fix in result["verdict"].get("fixes", [])
                   if str(fix).strip()]
    if judge_fixes and not reasons:
        reasons.append(f"{len(judge_fixes)} judge fixes")
    return reasons


def judge_passes(verdict):
    scores = verdict.get("scores", {})
    try:
        return all(float(scores.get(name, 0)) >= threshold
                   for name, threshold in JUDGE_THRESHOLDS.items())
    except (TypeError, ValueError):
        return False


def evaluate_story(story, age, request, premise):
    verdict = judge_story(story, age, request, premise)
    fixes = [str(fix).strip() for fix in verdict.get("fixes", [])
             if str(fix).strip()]
    fixes += sanity_checks(story)
    return {
        "story": story,
        "verdict": verdict,
        "score": average(verdict.get("scores", {})),
        "fixes": fixes,
        "passed": judge_passes(verdict) and not fixes,
    }


# pipeline
def make_story(request, verbose=False, age=None, use_memory=True):
    info = categorize(request)
    if age is None:
        try:
            age = int(info.get("age", 7))
        except (TypeError, ValueError):
            age = 7
    age = max(5, min(10, age))

    category = info.get("category", "animal_friend")
    if category not in CATEGORIES:
        category = "animal_friend"
    safe_request = str(info.get("safe_request", request))
    premise = add_twist(safe_request)
    memory = recall_memory() if use_memory else None

    if verbose:
        print(f"[category: {category} | age: {age}]")
        print(f"[premise: {premise}]")
        if memory:
            print(f"[remembering: {memory.get('name')}]")

    story = tell_story(premise, category, age, memory)
    story = ensure_length(story, age, verbose)

    if len(length_problems(story)) >= 2:
        if verbose:
            print("[structurally broken draft - regenerating once]")
        retry = ensure_length(
            tell_story(premise, category, age, memory), age, verbose
        )
        if len(length_problems(retry)) < len(length_problems(story)):
            story = retry

    history = []
    best = None
    first_score = None

    for round_number in range(MAX_REVISIONS + 1):
        current = evaluate_story(story, age, safe_request, premise)
        current["round"] = round_number
        history.append(current)

        if first_score is None:
            first_score = current["score"]
        if verbose:
            print(f"[judge {round_number}: {current['score']}/10, "
                  f"passed={current['passed']}, "
                  f"{len(current['fixes'])} fixes]")

        current_rank = (current["passed"], -len(current["fixes"]),
                        current["score"])
        if best is None or current_rank > \
                (best["passed"], -len(best["fixes"]), best["score"]):
            best = current

        if current["passed"]:
            break
        if round_number == MAX_REVISIONS or not current["fixes"]:
            break

        story = revise_story(story, current["fixes"], age)
        story = ensure_length(story, age, verbose)

    return {
        "story": best["story"],
        "score": best["score"],
        "first_score": first_score,
        "passed": best["passed"],
        "age": age,
        "category": category,
        "safe_request": safe_request,
        "premise": premise,
        "verdict": best["verdict"],
        "history": history,
    }


def apply_feedback(result, feedback, verbose=False):
    age = result["age"]
    story = revise_story(result["story"], [feedback], age)
    story = ensure_length(story, age, verbose)
    current = evaluate_story(
        story, age, result["safe_request"], result["premise"]
    )

    if not current["passed"] and current["fixes"]:
        story = revise_story(story, current["fixes"], age)
        story = ensure_length(story, age, verbose)
        current = evaluate_story(
            story, age, result["safe_request"], result["premise"]
        )

    updated = dict(result)
    updated.update({
        "story": current["story"],
        "score": current["score"],
        "passed": current["passed"],
        "verdict": current["verdict"],
        "history": result["history"] + [current],
    })
    return updated


def print_story(story, read_aloud=False):
    if not read_aloud:
        print(f"\n{story}\n")
        return

    parts = paragraphs(story)
    print()
    for index, part in enumerate(parts):
        print(part + "\n")
        time.sleep(0.5 + 1.5 * index / max(1, len(parts) - 1))


def main():
    parser = argparse.ArgumentParser(
        description="MoonDial - Your Bedtime storyteller"
    )
    parser.add_argument("--read-aloud", action="store_true")
    args = parser.parse_args()

    request = input("What kind of story do you want to hear? ").strip()
    if not request:
        request = ("A story about George and his best friend Stuart, "
                   "who happens to be a mouse.")

    age = None
    age_input = input(
        "How old are you? (5-10, Enter to skip) "
    ).strip()
    if age_input.isdigit():
        age = max(5, min(10, int(age_input)))

    print("\n Spinning the dial to write your story...")
    result = make_story(request, verbose=True, age=age)
    print_story(result["story"], args.read_aloud)
    print(f"(score: {result['score']}/10 | passed: {result['passed']})")

    while True:
        feedback = input(
            "\nAnything you would like to add in our story? "
            "(Press enter if perfect) "
        ).strip()
        if not feedback:
            break
        result = apply_feedback(result, feedback, verbose=True)
        print_story(result["story"], args.read_aloud)
        print(f"(score: {result['score']}/10 | passed: {result['passed']})")

    try:
        save_memory(result["story"])
    except Exception:
        pass

    print("\nSweet dreams, Sleep tight!")

if __name__ == "__main__":
    main()
