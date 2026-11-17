import os
import random
import tempfile
import unittest

from synthetic_data_generator.labeled import fill, generate_labeled, load_jsonl, noisy, save_jsonl, split

SPEC = {
    "slots": {"project": ["Atlas", "Nimbus", "Orbita"], "time": ["a las ocho", "mañana", "el lunes"]},
    "labels": {
        "project": [{"t": "arregla el test de {project}"}, {"t": "haz push en {project}"}, {"t": "sigue", "context": ["active_project"]}],
        "general": [{"t": "pon una alarma {time}"}, {"t": "¿qué tengo {time}?"}, {"t": "sigue", "context": ["recent_general"]}],
    },
}


class TestLabeled(unittest.TestCase):
    def test_generation_is_balanced_deduplicated_and_deterministic(self) -> None:
        a = generate_labeled(SPEC, 6, seed=1)
        b = generate_labeled(SPEC, 6, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(sum(s.label == "project" for s in a), sum(s.label == "general" for s in a))
        self.assertEqual(len({s.text for s in a}), len(a))
        sigue = [s for s in a if s.template == "sigue"]
        self.assertTrue(all(s.context for s in sigue))

    def test_grammar_limits_are_respected(self) -> None:
        tiny = {"labels": {"x": [{"t": "hola"}]}}
        self.assertLessEqual(len(generate_labeled(tiny, 50, noise=0.0)), 1)

    def test_unknown_slot(self) -> None:
        with self.assertRaises(KeyError):
            fill("{nope}", {}, random.Random(0))

    def test_noise_keeps_meaning_words(self) -> None:
        rng = random.Random(3)
        out = [noisy("¿Qué tengo mañana?", rng, 1.0) for _ in range(40)]
        self.assertTrue(all(o and "?" not in o and o == o.lower() for o in out))
        self.assertGreater(sum("tengo" in o for o in out), 20)  # a typo hits one word at most

    def test_split_by_template_holds_out_whole_templates(self) -> None:
        samples = generate_labeled(SPEC, 30, seed=2)
        train, ev = split(samples, 0.34, seed=0)
        key = lambda s: (s.label, s.template, tuple(s.context))  # noqa: E731
        self.assertFalse({key(s) for s in train} & {key(s) for s in ev})
        self.assertEqual({s.label for s in train}, {"project", "general"})
        self.assertTrue(ev)

    def test_jsonl_roundtrip(self) -> None:
        samples = generate_labeled(SPEC, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.jsonl")
            save_jsonl(samples, path)
            self.assertEqual(load_jsonl(path), samples)


if __name__ == "__main__":
    unittest.main()
