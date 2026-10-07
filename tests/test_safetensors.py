"""
test_safetensors.py — self-test for the safetensors audit path.

There is no safetensors library in this repo and no safetensors file on the box,
so every specimen here is synthesized: a valid file built byte-by-byte, then
variants that break one invariant each. The assertions mirror the GGUF template
tests — tampered specimens must be caught, the honest reference must pass clean.

Two invariants matter most:
  * per-tensor hashes must match an independent SHA-256 of the raw tensor bytes
    (proves the offsets we read are the real ones, not the header's claims);
  * a file whose header disagrees with its own bytes, its length, or its index
    must be caught, because that disagreement is what tampering looks like.

Run from the repo root:  python3 tests/test_safetensors.py
Exit 0 = all pass.
"""
import contextlib, hashlib, io, json, struct, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from model_audit import Report, audit

F16 = lambda n: b"\x00\x01" * n   # n fp16 elements, arbitrary but stable bytes


def build(tensors, meta=None, trailing=b"", header_extra=None, pad=True, offsets=None):
    """Synthesize a safetensors file. tensors: [(name, dtype, shape, raw_bytes)]."""
    data, entries = b"", {}
    for i, (name, dtype, shape, raw) in enumerate(tensors):
        begin, end = (offsets[i] if offsets else (len(data), len(data) + len(raw)))
        entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}
        data += raw
    if meta is not None:
        entries["__metadata__"] = meta
    if header_extra:
        entries.update(header_extra)
    hdr = json.dumps(entries).encode()
    if pad:
        hdr += b" " * ((8 - (8 + len(hdr)) % 8) % 8)
    return struct.pack("<Q", len(hdr)) + hdr + data + trailing


def honest_tensors():
    """A tiny but plausible transformer shard, HF naming."""
    return [("model.embed_tokens.weight", "BF16", [4, 4], F16(16)),
            ("model.layers.0.self_attn.q_proj.weight", "BF16", [4, 4], F16(16)),
            ("model.layers.0.self_attn.k_proj.weight", "BF16", [4, 4], F16(16)),
            ("model.layers.0.self_attn.v_proj.weight", "BF16", [4, 4], F16(16)),
            ("model.layers.0.self_attn.o_proj.weight", "BF16", [4, 4], F16(16))]


BENIGN_TEMPLATE = """{%- for m in messages -%}
{{ m['role'] }}: {{ m['content'] }}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ 'assistant' }}:{%- endif -%}
"""


def run(path, **kw):
    """Audit silently (the scanner prints templates verbatim) and return the Report."""
    rep = Report()
    with contextlib.redirect_stdout(io.StringIO()):
        audit(Path(path), rep, **kw)
    return rep


def msgs(rep, sev=None):
    return [m for s, _, m in rep.items if sev is None or s == sev]


class SAFETENSORS_BASE(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="model-audit-st-")
        self.dir = Path(self.tmp.name)
        self.file = self.dir / "model.safetensors"
        self.write_honest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_honest(self, **build_kw):
        self.file.write_bytes(build(honest_tensors(), meta={"format": "pt"}, **build_kw))
        (self.dir / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "torch_dtype": "bfloat16", "transformers_version": "4.44.0"}))
        (self.dir / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": BENIGN_TEMPLATE, "eos_token": "</s>"}))
        return self.file


class TestHonestReference(SAFETENSORS_BASE):
    def test_benign_file_passes_clean(self):
        rep = run(self.file)
        self.assertEqual([m for m in msgs(rep, "CRIT")], [],
                         "honest safetensors produced critical findings")
        self.assertEqual([m for m in msgs(rep, "WARN")], [],
                         "honest safetensors produced warnings")
        self.assertIn("format=safetensors", msgs(rep))

    def test_tensor_census_and_identity(self):
        rep = run(self.file)
        self.assertIn("5 tensors total", msgs(rep))
        self.assertTrue(any("80 parameters" in m for m in msgs(rep)),
                        "parameter count not reported")
        self.assertTrue(any(m.startswith("architecture=LlamaForCausalLM") for m in msgs(rep)),
                        "architecture from config.json not surfaced for the report id card")

    def test_per_tensor_hashes_match_independent_sha256(self):
        """The offsets we hashed must be the real ones — verify against the raw bytes."""
        rep = run(self.file, do_hashes=True)
        base = json.loads((self.dir / "model.safetensors.tensorhashes.json").read_text())
        self.assertEqual(base["format"], "safetensors")
        for name, dtype, shape, raw in honest_tensors():
            self.assertEqual(base["tensors"][name], hashlib.sha256(raw).hexdigest(),
                             f"tensor hash for {name} does not match its raw bytes")

    def test_template_sidecars_are_scanned(self):
        rep = run(self.file)
        self.assertTrue(any("template sidecar: tokenizer_config.json" in m for m in msgs(rep)),
                        "sidecar template not reported as scanned")

    def test_format_sniffing_does_not_claim_a_gguf(self):
        """A GGUF must still route to the gguf path (magic wins over 'looks like JSON')."""
        gguf = self.dir / "tiny.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)
        rep = run(gguf)
        self.assertIn("format=gguf", msgs(rep))


class TestStructuralTampering(SAFETENSORS_BASE):
    def test_trailing_data_is_flagged(self):
        self.file.write_bytes(build(honest_tensors(), trailing=b"hidden payload"))
        rep = run(self.file)
        self.assertTrue(any("trailing data" in m for m in msgs(rep, "WARN")),
                        "bytes appended after the last tensor went unreported")

    def test_header_size_contradiction_is_critical(self):
        """Header claims 64 bytes for a tensor whose dtype/shape imply 8."""
        tensors = honest_tensors()
        self.file.write_bytes(build(tensors, offsets=[(0, 64)] + [(64 + 16 * i, 64 + 16 * (i + 1))
                                                                for i in range(len(tensors) - 1)]))
        rep = run(self.file)
        self.assertTrue(any(m.startswith("tensor size:") for m in msgs(rep, "CRIT")),
                        "a tensor whose declared size contradicts its shape was not caught")

    def test_truncated_file_is_critical(self):
        blob = build(honest_tensors())
        self.file.write_bytes(blob[:-32])
        rep = run(self.file)
        self.assertTrue(any("past the end of the file" in m for m in msgs(rep, "CRIT")),
                        "truncated data region not caught")

    def test_absurd_header_length_is_critical(self):
        self.file.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
        rep = run(self.file)
        self.assertTrue(any("header unreadable" in m for m in msgs(rep, "CRIT")))

    def test_header_longer_than_file_is_critical(self):
        self.file.write_bytes(struct.pack("<Q", 4096) + b"{}" + b"\x00" * 16)
        rep = run(self.file)
        self.assertTrue(any("header unreadable" in m for m in msgs(rep, "CRIT")))

    def test_non_json_header_is_critical(self):
        self.file.write_bytes(struct.pack("<Q", 16) + b"not json at all" + b"\x00" * 64)
        rep = run(self.file)
        self.assertTrue(any("not valid JSON" in m or "bad magic" in m for m in msgs(rep, "CRIT")))

    def test_undeclared_gap_is_warned(self):
        tensors = honest_tensors()
        offs = [(0, 32)] + [(32 + 32 * i + 8, 32 + 32 * (i + 1) + 8) for i in range(len(tensors) - 1)]
        self.file.write_bytes(build(tensors, offsets=offs))
        rep = run(self.file)
        self.assertTrue(any("undeclared gap" in m for m in msgs(rep, "WARN")),
                        "an unaccounted-for region inside the file went unreported")

    def test_overlapping_tensors_are_critical(self):
        tensors = honest_tensors()
        offs = [(0, 32), (16, 48)] + [(64 + 32 * i, 64 + 32 * (i + 1)) for i in range(len(tensors) - 2)]
        self.file.write_bytes(build(tensors, offsets=offs))
        rep = run(self.file)
        self.assertTrue(any("tensor layout:" in m for m in msgs(rep, "CRIT")),
                        "overlapping tensor regions not caught")

    def test_empty_tensor_is_warned(self):
        self.file.write_bytes(build([("model.embed_tokens.weight", "F32", [0], b"")]
                                    + honest_tensors()[1:]))
        rep = run(self.file)
        self.assertTrue(any("is empty" in m for m in msgs(rep, "WARN")))


class TestSidecarAttacks(SAFETENSORS_BASE):
    """The template is the cheapest thing to swap — it is not in the weights file."""

    def test_hostile_chat_template_jinja_is_caught(self):
        (self.dir / "chat_template.jinja").write_text(
            (REPO / "templates" / "tampered" / "01-ssti-class-escape.j2").read_text())
        rep = run(self.file)
        crits_or_warns = msgs(rep, "CRIT") + msgs(rep, "WARN")
        self.assertTrue(any("dunder traversal" in m for m in crits_or_warns),
                        "hostile .jinja sidecar passed the scan")
        self.assertTrue(any(m.startswith("chat_template.jinja: ") for m in crits_or_warns),
                        "findings from the sidecar are not attributed to the sidecar file")

    def test_hostile_template_inside_tokenizer_config_is_caught(self):
        (self.dir / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": (REPO / "templates" / "tampered" / "03-invisible-instructions.j2").read_text()}))
        rep = run(self.file)
        self.assertTrue(any("zero-width" in m and "tokenizer_config.json" in m for m in msgs(rep, "CRIT")),
                        "zero-width payload hidden in tokenizer_config.json was not caught")

    def test_template_list_entries_are_all_scanned(self):
        (self.dir / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": [BENIGN_TEMPLATE,
                              (REPO / "templates" / "tampered" / "05-attribute-traversal.j2").read_text()]}))
        rep = run(self.file)
        self.assertTrue(any("chat_template[1]" in m and ("dunder" in m or "attr" in m)
                            for m in msgs(rep)),
                        "second template in a chat_template list was not scanned")

    def test_conflicting_template_sources_are_warned(self):
        (self.dir / "chat_template.jinja").write_text(BENIGN_TEMPLATE)
        (self.dir / "tokenizer_config.json").write_text(json.dumps({"chat_template": BENIGN_TEMPLATE.strip()}))
        rep = run(self.file)
        self.assertTrue(any("they differ" in m for m in msgs(rep, "WARN")),
                        "two disagreeing template sources went unreported")

    def test_auto_map_is_warned(self):
        (self.dir / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "auto_map": {"AutoModel": "modeling_custom.CustomModel"}}))
        rep = run(self.file)
        self.assertTrue(any("auto_map" in m for m in msgs(rep, "WARN")),
                        "config.json declaring remote code loading went unreported")

    def test_shipped_script_is_warned(self):
        (self.dir / "convert.py").write_text("# helper shipped next to the weights\n")
        rep = run(self.file)
        self.assertTrue(any("executable/script shipped alongside" in m for m in msgs(rep, "WARN")))

    def test_missing_template_sidecar_is_informational_only(self):
        (self.dir / "tokenizer_config.json").unlink()
        rep = run(self.file)
        self.assertEqual(msgs(rep, "CRIT"), [], "missing sidecar template is not a critical finding")
        self.assertTrue(any("no chat template sidecar" in m for m in msgs(rep, "INFO")))


class TestIndexConsistency(SAFETENSORS_BASE):
    def write_index(self, weight_map, total_size):
        (self.dir / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total_size}, "weight_map": weight_map}))

    def test_single_shard_index_agreeing_is_clean(self):
        names = [t[0] for t in honest_tensors()]
        self.write_index({n: "model.safetensors" for n in names}, 5 * 32)
        rep = run(self.file)
        self.assertEqual([m for m in msgs(rep, "CRIT")], [])

    def test_index_total_size_mismatch_is_critical(self):
        names = [t[0] for t in honest_tensors()]
        self.write_index({n: "model.safetensors" for n in names}, 999)
        rep = run(self.file)
        self.assertTrue(any("total_size" in m for m in msgs(rep, "CRIT")),
                        "index total_size disagreeing with the file was not caught")

    def test_tensor_routed_to_this_shard_but_absent_is_critical(self):
        names = [t[0] for t in honest_tensors()]
        self.write_index({n: "model.safetensors" for n in names} |
                         {"model.layers.9.mlp.down_proj.weight": "model.safetensors"}, 5 * 32)
        rep = run(self.file)
        self.assertTrue(any("absent from it" in m for m in msgs(rep, "CRIT")),
                        "a tensor the index says lives here but does not was not caught")

    def test_tensor_present_but_unlisted_is_warned(self):
        self.write_index({"model.embed_tokens.weight": "model.safetensors"}, 32)
        rep = run(self.file)
        self.assertTrue(any("not listed" in m for m in msgs(rep, "WARN")),
                        "a tensor added after indexing went unreported")


class TestCrossFormatDiff(unittest.TestCase):
    """GGUF and safetensors name the same weights differently, so a per-name diff is
    not meaningful across formats. The content match must carry it instead — and it
    must never emit the 'different model' CRIT for a legitimate conversion."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="model-audit-diff-")
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def baseline(self, name, fmt, tensors, shapes):
        p = self.dir / name
        p.write_text(json.dumps({"file_sha256": "0" * 64, "format": fmt,
                                 "shapes": shapes, "tensors": tensors}))
        return str(p)

    def run_diff(self, a, b):
        from model_audit import diff_baselines
        rep = Report()
        with contextlib.redirect_stdout(io.StringIO()):
            diff_baselines(a, b, rep)
        return rep

    def test_lossless_conversion_matches_every_blob_by_content(self):
        h = {f"t{i}": hashlib.sha256(bytes([i]) * 8).hexdigest() for i in range(3)}
        st = self.baseline("a.json", "safetensors", {f"model.layers.{i}.self_attn.q_proj.weight": v
                                                     for i, v in enumerate(h.values())},
                           {f"model.layers.{i}.self_attn.q_proj.weight": [8, 1] for i in range(3)})
        gg = self.baseline("b.json", "gguf", {f"blk.{i}.attn_q.weight": v for i, v in enumerate(h.values())},
                           {f"blk.{i}.attn_q.weight": [8, 1] for i in range(3)})
        rep = self.run_diff(st, gg)
        self.assertTrue(any(m.startswith("content match: all 3 weight blobs") for m in msgs(rep, "OK")),
                        "a lossless conversion was not recognised as content-identical")
        self.assertEqual(msgs(rep, "CRIT"), [], "legitimate cross-format conversion raised CRIT")

    def test_quantized_conversion_does_not_cry_wolf(self):
        st = self.baseline("a.json", "safetensors", {f"model.layers.{i}.mlp.down_proj.weight":
                            hashlib.sha256(b"fp%i" % i).hexdigest() for i in range(4)},
                           {f"model.layers.{i}.mlp.down_proj.weight": [4096, 11008] for i in range(4)})
        gg = self.baseline("b.json", "gguf", {f"blk.{i}.ffn_down.weight": hashlib.sha256(b"q4k%i" % i).hexdigest()
                                              for i in range(4)},
                           {f"blk.{i}.ffn_down.weight": [4096, 43] for i in range(4)})
        rep = self.run_diff(st, gg)
        self.assertEqual(msgs(rep, "CRIT"), [], "a quantized conversion produced a false critical")
        self.assertTrue(any("content match: 0 of 4" in m for m in msgs(rep, "INFO")))
        self.assertFalse([m for m in msgs(rep, "WARN") if "tensor only in" in m],
                         "cross-format diff emitted a per-tensor warning flood")

    def test_transposed_dims_still_match_by_content(self):
        """ggml stores 2-D dims reversed: shape must not gate a byte-identical match."""
        st = self.baseline("a.json", "safetensors", {"model.layers.0.mlp.up_proj.weight": "d" * 64},
                           {"model.layers.0.mlp.up_proj.weight": [4096, 11008]})
        gg = self.baseline("b.json", "gguf", {"blk.0.ffn_up.weight": "d" * 64},
                           {"blk.0.ffn_up.weight": [11008, 4096]})
        rep = self.run_diff(st, gg)
        self.assertTrue(any(m.startswith("content match: all 1 weight blob") for m in msgs(rep, "OK")),
                        "a lossless conversion with reversed dims was not matched by content")
        self.assertTrue(any("shape agreement: 1 of the 1" in m for m in msgs(rep, "INFO")))

    def test_same_format_diff_still_classifies_changes(self):
        a = self.baseline("a.json", "gguf", {"blk.0.ffn_down.weight": "a" * 64},
                          {"blk.0.ffn_down.weight": [10, 1]})
        b = self.baseline("b.json", "gguf", {"blk.0.ffn_down.weight": "b" * 64},
                          {"blk.0.ffn_down.weight": [10, 1]})
        rep = self.run_diff(a, b)
        self.assertTrue(any("TENSOR CONTENT CHANGED" in m for m in msgs(rep, "CRIT")),
                        "same-format content change must stay critical")


class TestUploaderManifest(unittest.TestCase):
    """A repo's own checksum list is a consistency check, not a safety claim: agreeing
    with it proves nothing, disagreeing with it is exactly the drift a swap leaves."""

    H = "a" * 64

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="model-audit-man-")
        self.dir = Path(self.tmp.name)
        self.file = self.dir / "model.gguf"
        self.file.write_bytes(b"weights")

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, text, name="MANIFEST.txt"):
        (self.dir / name).write_text(text)

    def check(self, sha):
        from model_audit import check_manifest
        rep = Report()
        check_manifest(self.file, sha, rep)
        return rep

    def test_no_manifest_is_silent(self):
        self.assertEqual(self.check(self.H).items, [], "absent manifest must add no findings")

    def test_matching_entry_is_ok(self):
        self.manifest(f"{self.H}  model.gguf\n")
        rep = self.check(self.H)
        self.assertTrue(any(m.startswith("SHA-256 matches the uploader's own entry") for m in msgs(rep, "OK")))
        self.assertEqual(msgs(rep, "CRIT"), [])

    def test_mismatch_under_own_name_is_critical(self):
        self.manifest(f"{'b' * 64}  model.gguf\n")
        rep = self.check(self.H)
        self.assertTrue(any(m.startswith("MANIFEST MISMATCH") for m in msgs(rep, "CRIT")),
                        "a file that disagrees with the uploader's own hash list must be critical")

    def test_renamed_file_is_reported_as_content_unchanged(self):
        self.manifest(f"{self.H}  Gemma-v2-Q4_K_M.gguf\n")
        rep = self.check(self.H)
        self.assertTrue(any("renamed since the manifest was written" in m for m in msgs(rep, "INFO")))

    def test_unlisted_file_is_informational(self):
        self.manifest(f"{'c' * 64}  other.gguf\n")
        rep = self.check(self.H)
        self.assertTrue(any("not listed in it" in m for m in msgs(rep, "INFO")))
        self.assertEqual([m for m in msgs(rep, "CRIT") + msgs(rep, "WARN")], [])

    def test_sha256sum_style_and_asterisk_markers_parse(self):
        self.manifest(f"{self.H}  *model.gguf\n")
        self.assertTrue(any(m.startswith("SHA-256 matches") for m in msgs(self.check(self.H), "OK")))

    def test_end_to_end_through_audit_on_safetensors(self):
        """The manifest check must fire on the safetensors path too, not just GGUF."""
        st = self.dir / "model.safetensors"
        st.write_bytes(build(honest_tensors()))
        sha = hashlib.sha256(st.read_bytes()).hexdigest()
        self.manifest(f"{sha}  model.safetensors\n")
        rep = run(st)
        self.assertTrue(any(m.startswith("SHA-256 matches the uploader's own entry") for m in msgs(rep, "OK")))


if __name__ == "__main__":
    unittest.main(verbosity=2)