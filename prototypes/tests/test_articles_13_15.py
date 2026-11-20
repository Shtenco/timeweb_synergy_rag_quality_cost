import unittest

from prototypes.scenarios.article13 import retain_facts
from prototypes.scenarios.article14 import diagnose_pipeline
from prototypes.scenarios.article15 import minimum_sample_size, wilson_interval


class LateScenarioTests(unittest.TestCase):
    def test_13_memory_window_drops_old_facts_and_keeps_recent_ones(self):
        retained = retain_facts(["old", "middle", "recent"], capacity=2)
        self.assertEqual(retained, ["middle", "recent"])

    def test_14_reports_retrieval_and_answer_support_separately(self):
        result = diagnose_pipeline(
            required_doc_ids={"a", "b"}, retrieved_doc_ids={"a", "b"},
            claims=["claim-1"], supported_claim_ids=set(),
        )
        self.assertEqual(result["retrieval_recall"], 1.0)
        self.assertEqual(result["answer_support_rate"], 0.0)

    def test_15_wilson_interval_contains_observed_proportion(self):
        low, high = wilson_interval(successes=9, trials=10)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)
        self.assertEqual(minimum_sample_size(margin=0.05), 385)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            wilson_interval(successes=0, trials=0)
        with self.assertRaisesRegex(ValueError, "between zero and trials"):
            wilson_interval(successes=11, trials=10)


if __name__ == "__main__":
    unittest.main()
