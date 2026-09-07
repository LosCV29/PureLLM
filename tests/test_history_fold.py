"""Standalone unit tests for utils/history.py — run: python -m unittest tests.test_history_fold
Loads the module by path so no Home Assistant install is needed."""
import importlib.util, pathlib, unittest

_p = pathlib.Path(__file__).resolve().parents[1] / "custom_components/purellm/utils/history.py"
_spec = importlib.util.spec_from_file_location("purellm_history", _p)
history = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(history)
fold = history.fold_history_into_user


class FoldTests(unittest.TestCase):
    def test_no_history_returns_none(self):
        self.assertIsNone(fold(None, "Yes, add peas."))
        self.assertIsNone(fold([], "Yes, add peas."))

    def test_non_string_content_returns_none(self):
        h = [{"role": "user", "content": "x"}, {"role": "assistant", "content": [{"type": "text", "text": "y"}]}]
        self.assertIsNone(fold(h, "z"))

    def test_folds_pair_into_single_user_message(self):
        h = [{"role": "user", "content": "Add tomatoes to the costco list."},
             {"role": "assistant", "content": "Added tomatoes to the Costco list. Anything else?  "}]
        out = fold(h, "Yes, add taco shells.")
        self.assertEqual(out, '[Context: the user said "Add tomatoes to the costco list." and you replied '
                              '"Added tomatoes to the Costco list. Anything else?"]\nThe user now says: Yes, add taco shells.')

    def test_list_name_context_survives(self):
        h = [{"role": "user", "content": "Add paper towels to the costco list."},
             {"role": "assistant", "content": "Added paper towels to the Costco list. Anything else?"}]
        self.assertIn("costco", fold(h, "Yes, add rotisserie chicken.").lower())

    def test_no_assistant_role_message_is_produced(self):
        # The whole point: the model must never see a replayed assistant text turn.
        h = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        out = fold(h, "c")
        self.assertIsInstance(out, str)
        self.assertTrue(out.endswith("The user now says: c"))


if __name__ == "__main__":
    unittest.main()
