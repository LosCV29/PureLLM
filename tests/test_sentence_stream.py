"""Standalone unit tests for utils/sentence_stream.py — run: python -m unittest tests.test_sentence_stream
Loads the module by path so no Home Assistant install is needed."""
import importlib.util, pathlib, re, unittest

_p = pathlib.Path(__file__).resolve().parents[1] / "custom_components/purellm/utils/sentence_stream.py"
_spec = importlib.util.spec_from_file_location("purellm_sentence_stream", _p)
mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mod)

GARBLED = "Sorry, I didn't catch that — can you repeat it?"
EMOJI = re.compile("[\U0001F000-\U0001FAFF]")


def make(max_chars=1200):
    return mod.SpokenSentenceStreamer(max_chars=max_chars, garbled_reply=GARBLED, emoji_re=EMOJI)


def run(s, tokens):
    out = []
    for t in tokens:
        out += s.feed(t)
    return out, s.finish()


class SentenceStreamTests(unittest.TestCase):
    def test_releases_each_sentence_as_it_completes(self):
        s = make()
        self.assertEqual(s.feed("It's currently 89 degrees"), [])
        self.assertEqual(s.feed(" with light rain. Raining"), ["It's currently 89 degrees with light rain. "])
        self.assertEqual(s.feed(" now."), [])
        self.assertEqual(s.finish(), ["Raining now."])

    def test_spoken_text_is_identical_to_input(self):
        text = "The front door is locked! Is that all? Yes. Done."
        streamed, tail = run(make(), list(text))
        self.assertEqual("".join(streamed + tail), text)

    def test_decimals_and_abbreviations_do_not_split_early(self):
        streamed, tail = run(make(), ["The shade is at 45.5 percent at 5:52 p.m.", " Anything else?"])
        self.assertEqual(streamed, ["The shade is at 45.5 percent at 5:52 p.m. "])
        self.assertEqual(tail, ["Anything else?"])

    def test_repeated_sentence_loop_is_dropped(self):
        streamed, tail = run(make(), ["Lights are on. ", "Lights are on. ", "Lights are on. ", "Done."])
        self.assertEqual(streamed + tail, ["Lights are on. ", "Done."])

    def test_emoji_stripped(self):
        streamed, _ = run(make(), ["Done 🎉. ", "Bye."])
        self.assertEqual(streamed, ["Done . "])

    def test_garble_first_collapses_to_canned_reply(self):
        streamed, tail = run(make(), ["Sorry, I didn't catch that. I can help with lights, music. "])
        self.assertEqual(streamed + tail, [GARBLED])

    def test_garble_after_speech_stops_without_canned(self):
        streamed, tail = run(make(), ["The door is locked. ", "Sorry, I didn't catch that. ", "More."])
        self.assertEqual(streamed + tail, ["The door is locked. "])

    def test_char_cap_stops_at_sentence_boundary(self):
        streamed, tail = run(make(max_chars=40), ["First sentence here. ", "Second sentence is long. ", "Third."])
        self.assertEqual(streamed + tail, ["First sentence here. "])
        self.assertTrue(make(max_chars=40).done is False)

    def test_angle_bracket_holds_everything_after(self):
        s = make()
        streamed, tail = run(s, ["Checking now. ", '<tool_call>{"name": "x"}', "</tool_call> ok. "])
        self.assertEqual(streamed, ["Checking now. "])
        self.assertEqual(tail, [])
        self.assertTrue(s.held)
        self.assertIn("<tool_call>", s.buffer)

    def test_nothing_released_without_a_boundary(self):
        s = make()
        self.assertEqual(s.feed("Single sentence with no trailing space."), [])
        self.assertFalse(s.emitted_any)


if __name__ == "__main__":
    unittest.main()
