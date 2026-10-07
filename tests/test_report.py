"""
test_report.py — self-test for the plain-language HTML report rendering.

render_file_block() turns a list of (severity, section, message) findings into the block
a reader sees. Two things it must never do, both of which have happened:

  * drop a finding because its message shape did not match a filter key (the id card's
    provenance rows compared "base_model" against "base_model " and silently rendered
    nothing, for both formats);
  * show internal section keys ("st-tensors", "st-header") to a non-technical reader.

Also: every CRITICAL/WARNING row must carry its plain-language explanation, and the
per-file identity card must state the format that was audited.

Run from the repo root:  python3 tests/test_report.py
"""
import sys, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from report import render_file_block, SECTION_LABELS, PLAIN


def st_items():
    """A finding list shaped exactly like a real safetensors audit."""
    return [
        ("INFO", "file", "/models/model.safetensors"),
        ("INFO", "file", "sha256=" + "a" * 64),
        ("INFO", "file", "size=1,000"),
        ("INFO", "file", "format=safetensors"),
        ("INFO", "st-header", "header 2,184 bytes, data section starts at 2,192, file 1,000,000"),
        ("INFO", "meta", "architecture=LlamaForCausalLM name='ex/model' file_type=bfloat16 (safetensors)"),
        ("INFO", "meta", "model_type = llama"),
        ("INFO", "meta", "torch_dtype = bfloat16"),
        ("INFO", "meta", "base_model = meta-llama/Llama-3.1-8B"),
        ("INFO", "tensors", "5 tensors total"),
        ("INFO", "st-tensors", "dtypes: BF16:5; 80 parameters in this file"),
        ("WARN", "st-tensors", "trailing data: 14 bytes after the last declared tensor"),
        ("INFO", "manifest", "MANIFEST.txt present (5 entries) but this file is not listed in it"),
        ("WARN", "sidecar", "auto_map declared in config.json: loading this model with trust_remote_code…"),
        ("INFO", "template", "template sidecar: chat_template.jinja (17,466 bytes)"),
    ]


class TestRenderFileBlock(unittest.TestCase):
    def setUp(self):
        self.html = render_file_block("model.safetensors", st_items())

    def test_identity_card_states_the_format(self):
        self.assertIn("<td>Format</td><td>safetensors</td>", self.html,
                      "the audited format must appear in the identity card")

    def test_provenance_rows_are_not_dropped(self):
        for row in ("model_type", "torch_dtype", "base_model"):
            self.assertIn(f"<td>{row}</td>", self.html,
                          f"identity card row {row!r} was dropped by the filter")

    def test_template_verdict_and_source_render(self):
        self.assertIn("Chat template", self.html)
        self.assertIn("<td>Template source</td>", self.html)
        self.assertIn("chat_template.jinja", self.html)

    def test_no_internal_section_keys_leak(self):
        for key in SECTION_LABELS:
            self.assertNotIn(f'<span class="sec">{key}</span>', self.html,
                             f"internal section key {key!r} shown to the reader")
        self.assertIn('<span class="sec">Weight structure</span>', self.html)

    def test_every_warning_row_has_a_plain_explanation(self):
        import re
        for row in re.findall(r'<tr class="(?:crit|warn)">.*?</tr>', self.html, re.S):
            self.assertIn('class="plain"', row, f"explanation missing for row: {row[:120]}")

    def test_fingerprint_and_size_do_not_appear_as_findings(self):
        self.assertNotIn("sha256=" + "a" * 64, self.html,
                         "raw identity lines must live in the header table, not the findings")

    def test_unknown_section_key_still_renders(self):
        out = render_file_block("x", [("WARN", "brand-new-check", "something happened")])
        self.assertIn("brand-new-check", out, "an unmapped section key must still render")


class TestPlainTranslations(unittest.TestCase):
    """AGENTS rule: every finding carries a non-technical translation. Pin the ones the
    safetensors/checksum/manifest work added."""

    def test_new_findings_have_translations(self):
        samples = [
            "header invalid: not valid JSON (…)", "tensor size: x is BF16 [4, 4], which implies 32 bytes…",
            "trailing data: 14 bytes after the last declared tensor", "undeclared gap: 8 bytes between tensors",
            "auto_map declared in config.json", "MANIFEST MISMATCH: this file is …",
            "no chat template sidecar in this directory", "index mismatch: metadata.total_size is …",
            "the gguf python package is not installed", "bad magic b'\\x00\\x01\\x02\\x03' — not a GGUF…",
        ]
        for msg in samples:
            hit = [k for k in PLAIN if msg.startswith(k) or k in msg[:40] or k in msg]
            self.assertTrue(hit, f"no plain-language translation matches: {msg!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
