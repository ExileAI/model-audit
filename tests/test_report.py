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
import contextlib, io, json, sys, tempfile, unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import report
from model_audit import Report, evidence_rubric
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
            "audit failed: unreadable", "input error: invalid baseline", "template invalid: expected text",
            "template coverage incomplete: no template source was available", "template sources for 'default': they differ",
            "tensor entry invalid dtype or shape: weight", "baseline refused: tensor structure was not fully validated",
            "baseline comparison incomplete: missing shape metadata", "TENSOR SHAPE CHANGED: weight",
            "TENSOR CONTENT CHANGED: weight", "comparing across formats or legacy unknown formats",
            "baseline exists: existing.json — pass --baseline-out to write a new snapshot",
            "empty tensor: weight is empty (F32 [0]) — legal in safetensors; inventory heuristic only",
            "normalized-hash candidates lack compatible typed normalization metadata",
        ]
        for msg in samples:
            hit = [k for k in PLAIN if msg.startswith(k) or k in msg[:40] or k in msg]
            self.assertTrue(hit, f"no plain-language translation matches: {msg!r}")


class TestConservativeExplanations(unittest.TestCase):
    def test_empty_tensor_and_output_conflict_are_not_format_failures(self):
        text = report.plain_for('WARN', 'st-tensors', 'empty tensor: weight is empty (F32 [0])')
        self.assertIn('legal in safetensors', text)
        self.assertNotIn('likely broken', text)
        text = report.plain_for('WARN', 'baseline', 'baseline exists: original.json')
        self.assertIn('preserved', text)
        self.assertIn('does not invalidate', text)

    def test_unknown_dtype_is_not_a_proven_contradiction(self):
        text = report.plain_for('WARN', 'st-tensors',
            "tensor size: weight dtype 'UNKNOWN' is not a known type — its byte size cannot be verified")
        self.assertIn('incomplete check', text)
        self.assertNotIn('header contradicts', text)
        self.assertNotIn('grafted', text)

    def test_other_name_hash_is_not_proven_rename_history(self):
        text = report.plain_for('INFO', 'manifest', 'renamed since the manifest was written: weights')
        self.assertIn('does not establish rename history', text)
        self.assertNotIn('harmless', text)


class TestEvidenceRubricRendering(unittest.TestCase):
    def test_four_domains_status_reason_scope_and_requested_coverage(self):
        rep = Report()
        for key in ("structure", "templates", "provenance", "baseline"):
            rep.observe(key, "PASS", f"{key} reason", f"{key} scope")
        rendered = render_file_block("model", [], evidence_rubric(rep))
        for key, label in report.RUBRIC_LABELS.items():
            self.assertIn(f'<th scope="row">{label}</th>', rendered)
            self.assertIn(f"{key} reason", rendered)
            self.assertIn(f"{key} scope", rendered)
        self.assertEqual(rendered.count("<td>PASS</td>"), 5)  # Four rows plus template card.
        self.assertIn("Requested coverage: COMPLETE", rendered)
        self.assertIn("<th>Status</th><th>Reason</th><th>Scope</th>", rendered)

    def test_required_unchecked_is_incomplete_and_not_clean(self):
        rep = Report()
        rep.observe("templates", "NOT CHECKED", "Template could not be read", "templates/*.jinja")
        rendered = render_file_block("model", [], evidence_rubric(rep))
        self.assertIn("Requested coverage: INCOMPLETE", rendered)
        self.assertIn("<td>NOT CHECKED</td>", rendered)
        self.assertNotIn("✅ CLEAN", rendered)
        self.assertNotIn("HOSTILE PATTERNS FOUND", rendered)

    def test_legacy_findings_do_not_claim_completed_template_scan(self):
        rendered = render_file_block("model", st_items())
        self.assertIn("<td>Chat template</td><td>NOT CHECKED</td>", rendered)
        self.assertNotIn("✅ CLEAN", rendered)

    def test_rubric_and_evidence_are_html_escaped(self):
        marker = '<script>alert("x")</script>&'
        rep = Report()
        for key in report.RUBRIC_LABELS:
            rep.observe(key, "PASS", marker, marker)
        rubric = evidence_rubric(rep)
        rubric["domains"]["templates"]["status"] = marker
        rubric["coverage"] = dict(status=marker, reason=marker, scope=marker)
        rendered = render_file_block(marker, [("WARN", "template", marker)], rubric)
        self.assertNotIn(marker, rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&amp;", rendered)

    def test_generic_audit_errors_have_explanation_and_honest_verdict(self):
        rendered = render_file_block("model", [("CRIT", "audit", "audit failed: unreadable")])
        self.assertIn('class="plain"', rendered)
        self.assertIn("not the intent", rendered)
        self.assertNotIn("this file is unsafe", rendered)
        self.assertNotIn("weights are safe data", " ".join(PLAIN.values()))


class TestReportGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outdir = Path(self.tmp.name) / "reports"

    @staticmethod
    def audit_with(severity):
        def audit(path, rep):
            rep.add(severity, "test", f"{path.name} observed finding")
            rep.observe("structure", "PASS", "Structure inspected", str(path))
        return audit

    def test_frozen_clock_generates_distinct_pairs_without_overwriting(self):
        frozen = datetime(2026, 10, 8, 12, 34, 56, tzinfo=timezone.utc)
        with patch.object(report, "datetime") as clock, patch.object(report, "_audit", self.audit_with("INFO")):
            clock.now.return_value = frozen
            first = report.generate([Path("first.gguf")], self.outdir)
            first_html = first.read_bytes()
            first_json = first.with_suffix(".json").read_bytes()
            second = report.generate([Path("second.gguf")], self.outdir)
        clock.now.assert_called_with(timezone.utc)
        self.assertIsInstance(first, Path)
        self.assertNotEqual(first.parent, second.parent)
        for output in (first, second):
            self.assertRegex(output.parent.name, r"^audit-report-20261008T123456Z-.+$")
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
        self.assertEqual(first.read_bytes(), first_html)
        self.assertEqual(first.with_suffix(".json").read_bytes(), first_json)
        self.assertIn("2026-10-08 12:34:56 UTC", first.read_text())

    def test_json_preserves_findings_and_adds_shared_rubric(self):
        with patch.object(report, "_audit", self.audit_with("WARN")):
            output = report.generate([Path("model.gguf")], self.outdir)
        rows = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(rows[0]["file"], "model.gguf")
        self.assertEqual(rows[0]["findings"], [dict(severity="WARN", section="test", message="model.gguf observed finding")])
        expected = Report()
        self.audit_with("WARN")(Path("model.gguf"), expected)
        self.assertEqual(rows[0]["rubric"], evidence_rubric(expected))

    def test_worst_severity_exit_codes_and_pair_exists_when_announced(self):
        for severities, expected in ((["INFO", "OK"], 0), (["INFO", "WARN"], 1),
                                     (["CRIT", "WARN", "INFO"], 2)):
            with self.subTest(severities=severities):
                def audit(path, rep):
                    rep.add(severities[int(path.name)], "test", "test finding")
                stdout = io.StringIO()
                real_print = print
                def checked_print(value, **kwargs):
                    if "file" not in kwargs:
                        path = Path(value)
                        self.assertTrue(path.is_file())
                        self.assertTrue(path.with_suffix(".json").is_file())
                    real_print(value, **kwargs)
                with patch.object(report, "_audit", audit), patch("builtins.print", checked_print), contextlib.redirect_stdout(stdout):
                    status = report.main([*[str(i) for i in range(len(severities))], "-o", str(self.outdir)])
                self.assertEqual(status, expected)
                self.assertEqual(len(stdout.getvalue().splitlines()), 1)

    def test_audit_exception_is_critical_and_coverage_incomplete(self):
        with patch.object(report, "_audit", side_effect=OSError("cannot read model")):
            output, status = report._generate([Path("missing.gguf")], self.outdir)
        self.assertEqual(status, 2)
        row = json.loads(output.with_suffix(".json").read_text())[0]
        self.assertEqual(row["findings"][0]["severity"], "CRIT")
        self.assertEqual(row["rubric"]["coverage"]["status"], "INCOMPLETE")
        self.assertIn("Requested coverage: INCOMPLETE", output.read_text())

    def test_second_open_failure_cleans_own_run_but_keeps_prior_pair(self):
        with patch.object(report, "_audit", self.audit_with("INFO")):
            prior = report.generate([Path("prior.gguf")], self.outdir)
        original = {p: p.read_bytes() for p in prior.parent.iterdir()}
        actual_open = Path.open
        def fail_json(path, mode="r", *args, **kwargs):
            if path.suffix == ".json" and mode == "x":
                raise OSError("simulated JSON open failure")
            return actual_open(path, mode, *args, **kwargs)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(report, "_audit", self.audit_with("INFO")), patch.object(Path, "open", fail_json), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = report.main(["next.gguf", "-o", str(self.outdir)])
        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Report generation failed", stderr.getvalue())
        self.assertEqual(list(self.outdir.iterdir()), [prior.parent])
        self.assertEqual({p: p.read_bytes() for p in prior.parent.iterdir()}, original)

    def test_partial_second_write_failure_removes_both_outputs(self):
        actual_open = Path.open
        class PartialFailure:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                self.stream.__enter__()
                return self
            def write(self, content):
                self.stream.write(content[:10])
                raise OSError("simulated disk full")
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
        def fail_json_write(path, mode="r", *args, **kwargs):
            stream = actual_open(path, mode, *args, **kwargs)
            return PartialFailure(stream) if path.suffix == ".json" and mode == "x" else stream
        with patch.object(report, "_audit", self.audit_with("INFO")), patch.object(Path, "open", fail_json_write):
            with self.assertRaisesRegex(OSError, "simulated disk full"):
                report.generate([Path("model.gguf")], self.outdir)
        self.assertEqual(list(self.outdir.iterdir()), [])

    def test_exclusive_open_does_not_overwrite_or_delete_existing_file(self):
        self.outdir.mkdir()
        run_dir = self.outdir / "collision"
        run_dir.mkdir()
        existing = run_dir / "audit-report.json"
        existing.write_text("earlier evidence")
        with patch.object(report, "_audit", self.audit_with("INFO")), patch.object(report.tempfile, "mkdtemp", return_value=str(run_dir)):
            with self.assertRaises(FileExistsError):
                report.generate([Path("model.gguf")], self.outdir)
        self.assertEqual(existing.read_text(), "earlier evidence")
        self.assertFalse((run_dir / "audit-report.html").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
