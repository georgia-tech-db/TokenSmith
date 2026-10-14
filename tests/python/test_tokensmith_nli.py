import unittest

from python_engine import tokensmith_nli as nli


PASSAGE = "A transaction preserves atomicity and durability across a database crash."


class ClaimSplittingTests(unittest.TestCase):
    def test_splits_sentences_and_drops_markdown_and_code(self):
        answer = (
            "**Overview**\n\n"
            "- Pinning keeps a page in memory. It stops eviction while in use.\n"
            "```cpp\nbufferManager.getPage(10);\n```\n"
            "Why does this matter?"
        )

        self.assertEqual(
            nli.split_claims(answer),
            ["Pinning keeps a page in memory.", "It stops eviction while in use."],
        )

    def test_checks_a_fragment_answer_together_with_its_question(self):
        self.assertEqual(
            nli.split_claims("November 11th, 2025.", "When is the project due?"),
            ["When is the project due? November 11th, 2025."],
        )

    def test_leaves_out_statements_about_the_context_itself(self):
        answer = (
            "There is no mention of a random scheduler in the provided context. "
            "The project covers FCFS, Round-Robin, and priority scheduling."
        )

        self.assertEqual(nli.split_claims(answer), ["The project covers FCFS, Round-Robin, and priority scheduling."])


class NamedTermTests(unittest.TestCase):
    def test_finds_acronyms_numbers_identifiers_and_mid_sentence_names(self):
        self.assertEqual(
            nli.named_terms("B+ trees were invented at IBM in 1985, per try_to_insert docs written in Java."),
            ["1985", "IBM", "Java", "try_to_insert"],
        )

    def test_ignores_the_capital_that_starts_a_sentence(self):
        self.assertEqual(nli.named_terms("Pinning keeps a page in memory."), [])

    def test_matches_terms_as_whole_words_regardless_of_case(self):
        self.assertTrue(nli.term_in("OLTP", "workloads called oltp systems"))
        self.assertFalse(nli.term_in("IO", "a ratio of reads"))


class CandidateSpanTests(unittest.TestCase):
    def test_prefers_a_span_that_names_what_the_claim_names(self):
        passages = [
            "Analysts scan billions of historical records. The first workload, many small transactions, is called OLTP."
        ]
        spans = nli.passage_spans(passages)
        words = [nli.span_words(text) for _, text in spans]
        idf = {word: 1.0 for group in words for word in group}

        chosen = nli.candidate_spans("OLTP scans billions of historical records.", spans, words, idf, ["OLTP"])

        self.assertIn("OLTP", spans[chosen[0]][1])


@unittest.skipUnless(nli.default_model_dir().exists(), "NLI model not downloaded; run npm run setup:python-runtime")
class ModelTests(unittest.TestCase):
    def verdict(self, answer, passages=(PASSAGE,), question=""):
        [claim] = nli.verify_claims(answer, list(passages), question=question)["claims"]
        return (
            max(passage["entailment"] for passage in claim["passages"]),
            max(passage["contradiction"] for passage in claim["passages"]),
        )

    def test_entails_the_claim_the_passage_makes(self):
        entailment, _ = self.verdict("A transaction preserves atomicity and durability.")
        self.assertGreater(entailment, 0.8)

    def test_contradicts_the_negated_claim_word_overlap_used_to_pass(self):
        entailment, contradiction = self.verdict("A transaction does not preserve atomicity and durability.")
        self.assertLess(entailment, 0.2)
        self.assertGreater(contradiction, 0.8)

    def test_does_not_entail_an_unrelated_fact(self):
        entailment, _ = self.verdict("Paris is the capital of France.")
        self.assertLess(entailment, 0.2)

    def test_never_entails_a_claim_naming_something_the_passages_do_not(self):
        result = nli.verify_claims("In 1985, transactions preserved atomicity and durability.", [PASSAGE])
        [claim] = result["claims"]
        self.assertEqual(claim["missingTerms"], ["1985"])
        self.assertEqual(max(passage["entailment"] for passage in claim["passages"]), 0.0)

    def test_reports_unavailable_instead_of_raising(self):
        result = nli.verify_answer({"answer": "Anything at all here.", "passages": [PASSAGE], "modelDir": "/nonexistent"})
        self.assertFalse(result["available"])
        self.assertEqual(result["claims"], [])


if __name__ == "__main__":
    unittest.main()
