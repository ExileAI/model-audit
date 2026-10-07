"""
test_scanner.py — self-test for the model-audit chat-template scanner.

Feeds every template in templates/tampered/ through the scanner and asserts it
fires; feeds every template in templates/benign/ and asserts it stays clean.
Doubles as a living demo of what the scanner catches.

Run from the repo root:  python3 tests/test_scanner.py
Exit 0 = all pass.
"""
import sys, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model_audit import Report, scan_template

# minimum findings a specimen must produce to prove detection
EXPECTED = {
    "01-ssti-class-escape.j2":      {"min_warnings": 2, "need_labels": ["python dunder traversal"]},
    "02-exfil-system-prompt.j2":    {"min_warnings": 1, "need_labels": ["template fabricates a tool call"]},
    "03-invisible-instructions.j2": {"min_crits": 1,    "need_labels": ["zero-width"]},
    "04-homoglyph-smuggle.j2":      {"min_warnings": 1, "need_labels": ["confusable"]},
    "05-attribute-traversal.j2":    {"min_warnings": 2, "need_labels": ["python dunder traversal"]},
}

def run_scanner(text):
    rep = Report()
    scan_template(text, rep)
    # silence the verbatim template print for test runs
    return rep

class TestTamperedSpecimens(unittest.TestCase):
    def test_each_specimen_is_detected(self):
        for fname, want in EXPECTED.items():
            with self.subTest(specimen=fname):
                path = REPO / "templates" / "tampered" / fname
                self.assertTrue(path.exists(), f"missing specimen: {fname}")
                rep = run_scanner(path.read_text())
                crits = [m for s, _, m in rep.items if s == "CRIT"]
                warns = [m for s, _, m in rep.items if s == "WARN"]
                labels = " | ".join(crits + warns)
                self.assertGreaterEqual(len(crits), want.get("min_crits", 0),
                                        f"{fname}: expected >= {want.get('min_crits',0)} CRIT, got {len(crits)}")
                self.assertGreaterEqual(len(warns), want.get("min_warnings", 0),
                                        f"{fname}: expected >= {want.get('min_warnings',0)} WARN, got {len(warns)}")
                for label in want["need_labels"]:
                    self.assertIn(label, labels, f"{fname}: expected finding containing {label!r}")

    def test_specimen_folder_matches_expectations(self):
        on_disk = {p.name for p in (REPO / "templates" / "tampered").glob("*.j2")}
        self.assertEqual(on_disk, set(EXPECTED),
                         "a specimen exists on disk without a test expectation (or vice versa)")

class TestBenignTemplates(unittest.TestCase):
    def test_benign_stays_clean(self):
        for path in (REPO / "templates" / "benign").glob("*.j2"):
            with self.subTest(template=path.name):
                rep = run_scanner(path.read_text())
                bad = [(s, m) for s, _, m in rep.items if s in ("CRIT", "WARN")]
                self.assertEqual(bad, [], f"{path.name}: clean template flagged: {bad}")

class TestScannerCore(unittest.TestCase):
    def test_realistic_toolcalling_template_not_flagged_as_hostile(self):
        # tool-calling templates legitimately mention files/tools; must not go CRIT
        tpl = """{%- for m in messages -%}
{%- if m['role'] == 'tool' -%}<tool_response>{{ m['content'] }}</tool_response>
{%- elif m['role'] == 'assistant' and m.get('tool_calls') -%}
{{ m['tool_calls'][0]['function']['name'] }}
{%- endif -%}{%- endfor -%}"""
        rep = run_scanner(tpl)
        crits = [m for s, _, m in rep.items if s == "CRIT"]
        self.assertEqual(crits, [], f"legitimate tool-calling template flagged CRIT: {crits}")

    def test_zero_width_detection(self):
        rep = run_scanner("{{ m['content'] }}\u200b\u200c")
        self.assertTrue(any(s == "CRIT" and "zero-width" in m for s, _, m in rep.items))

if __name__ == "__main__":
    unittest.main(verbosity=2)
