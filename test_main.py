import json
import unittest
import main

def paragraph(sentence: str, target_words: int = 70) -> str:
    words = sentence.split()
    repeats = (target_words + len(words) - 1) // len(words)
    return " ".join((words * repeats)[:target_words])

BAD_STORY = "\n\n".join(
    [
        paragraph("Milo found a bell."),
        paragraph("Milo followed its sound."),
        paragraph("Milo hurried through the meadow."),
        paragraph("Milo brought the bell home."),
        paragraph("Milo cheered and jumped!"),
    ]
)

GOOD_STORY = "\n\n".join(
    [
        paragraph("Milo found a quiet silver bell."),
        paragraph("Milo followed its gentle sound."),
        paragraph("Milo asked Moss for help."),
        paragraph("Together they carried the bell home."),
        paragraph("Milo rested in warm light and dreamed."),
    ]
)

ONE_PARAGRAPH_STORY = " ".join(["Milo rested quietly."] * 400)

class ScriptedModel:
    def __init__(self):
        self.judge_calls = 0
        self.revision_calls = 0
        self.judge_responses = []

    def __call__(self, prompt, temperature=0.8, max_tokens=1400):
        if "prepare bedtime-story requests" in prompt:
            return json.dumps(
                {
                    "category": "animal_friend",
                    "age": 6,
                    "safe_request": "a story about a mouse",
                    "was_softened": False,
                }
            )
        if "bedtime-story premise more original" in prompt:
            return json.dumps(
                {"premise": "a mouse whose bell rings near lost things"}
            )
        if "warm bedtime storyteller" in prompt:
            return BAD_STORY
        if "children's librarian evaluating" in prompt:
            if self.judge_responses:
                return json.dumps(self.judge_responses.pop(0))
            self.judge_calls += 1
            if "jumped!" in prompt:
                return json.dumps(
                    {
                        "scores": {
                            "request_fidelity": 9,
                            "age_appropriate": 9,
                            "story_arc": 6,
                            "calm_ending": 3,
                            "engaging": 7,
                        },
                        "fixes": ["End with a quiet sleepy image."],
                    }
                )
            return json.dumps(
                {
                    "scores": {
                        "request_fidelity": 9,
                        "age_appropriate": 9,
                        "story_arc": 8,
                        "calm_ending": 10,
                        "engaging": 8,
                    },
                    "fixes": [],
                }
            )
        if "Revise this bedtime story" in prompt:
            self.revision_calls += 1
            return GOOD_STORY
        if "Expand this bedtime story" in prompt:
            return GOOD_STORY
        if "Pick ONE small harmless element" in prompt:
            return json.dumps(
                {"name": "Silver Bell", "description": "A quiet lost bell."}
            )
        if "previous reply was not valid JSON" in prompt:
            return json.dumps({"corrected": True})
        raise AssertionError("unmatched prompt: " + prompt[:100])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.original_call_model = main.call_model
        self.original_recall_memory = main.recall_memory
        self.model = ScriptedModel()
        main.call_model = self.model
        main.recall_memory = lambda: None

    def tearDown(self):
        main.call_model = self.original_call_model
        main.recall_memory = self.original_recall_memory

    def test_pipeline_repairs_and_returns_passing_draft(self):
        result = main.make_story("a story about a mouse", use_memory=False)
        self.assertEqual(result["story"], GOOD_STORY)
        self.assertTrue(result["passed"])
        self.assertEqual(result["first_score"], 6.8)
        self.assertEqual(result["score"], 8.8)

    def test_stated_age_overrides_model_guess(self):
        result = main.make_story(
            "a story about a mouse", age=9, use_memory=False
        )
        self.assertEqual(result["age"], 9)

    def test_sanity_checks_find_loud_ending(self):
        problems = main.sanity_checks(BAD_STORY)
        self.assertTrue(any("exclamation" in problem for problem in problems))

    def test_sanity_checks_require_five_paragraphs(self):
        problems = main.sanity_checks(ONE_PARAGRAPH_STORY)
        self.assertTrue(any("exactly 5" in problem for problem in problems))

    def test_sanity_checks_pass_good_story(self):
        self.assertEqual(main.sanity_checks(GOOD_STORY), [])

    def test_short_story_triggers_expansion(self):
        original_tell_story = main.tell_story
        main.tell_story = lambda *args, **kwargs: "A tiny story."
        try:
            result = main.make_story("a story about a mouse", use_memory=False)
            self.assertEqual(result["story"], GOOD_STORY)
        finally:
            main.tell_story = original_tell_story

    def test_low_safety_score_cannot_be_averaged_away(self):
        verdict = {
            "scores": {
                "request_fidelity": 10,
                "age_appropriate": 5,
                "story_arc": 10,
                "calm_ending": 10,
                "engaging": 10,
            },
            "fixes": [],
        }
        self.assertFalse(main.judge_passes(verdict))

    def test_valid_draft_beats_higher_scoring_invalid_draft(self):
        self.model.judge_responses = [
            {
                "scores": {
                    "request_fidelity": 10,
                    "age_appropriate": 10,
                    "story_arc": 10,
                    "calm_ending": 8,
                    "engaging": 8,
                },
                "fixes": [],
            },
            {
                "scores": {
                    "request_fidelity": 8,
                    "age_appropriate": 9,
                    "story_arc": 8,
                    "calm_ending": 8,
                    "engaging": 7,
                },
                "fixes": [],
            },
        ]

        result = main.make_story("a story about a mouse", use_memory=False)
        self.assertEqual(result["story"], GOOD_STORY)
        self.assertTrue(result["passed"])
        self.assertLess(result["score"], result["first_score"])

    def test_judge_prompt_contains_request_for_fidelity(self):
        captured = {}

        def capture(prompt, temperature=0.8, max_tokens=1400):
            captured["prompt"] = prompt
            return json.dumps(
                {
                    "scores": {
                        "request_fidelity": 9,
                        "age_appropriate": 9,
                        "story_arc": 9,
                        "calm_ending": 9,
                        "engaging": 9,
                    },
                    "fixes": [],
                }
            )

        main.call_model = capture
        main.judge_story(
            GOOD_STORY,
            7,
            "George and Stuart the mouse learn to share",
            "George and Stuart share a moon-shaped blanket",
        )
        self.assertIn("George and Stuart the mouse", captured["prompt"])
        self.assertIn("request_fidelity", captured["prompt"])

    def test_json_extraction_handles_brace_inside_string(self):
        result = main.extract_json(
            'prefix {"message": "a } brace", "nested": {"ok": true}} suffix'
        )
        self.assertEqual(result["nested"]["ok"], True)

if __name__ == "__main__":
    unittest.main()
