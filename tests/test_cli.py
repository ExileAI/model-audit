"""Regression contract: conservative evidence, strict inputs and preserved outputs."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model_audit as m
from test_safetensors import build, BENIGN_TEMPLATE

H = lambda b: hashlib.sha256(b).hexdigest()


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.model = self.d / 'model.safetensors'
        self.model.write_bytes(build([('weight', 'F32', [1], b'\0' * 4)]))
        self.config = self.d / 'tokenizer_config.json'
        self.config.write_text(json.dumps({'chat_template': BENIGN_TEMPLATE}))

    def tearDown(self):
        self.tmp.cleanup()

    def audit(self, **kw):
        rep = m.Report()
        m.audit(self.model, rep, **kw)
        return rep

    def cli(self, *args, script='model_audit.py'):
        return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], capture_output=True, text=True)

    def raw_header(self, obj, data=b''):
        raw = obj.encode()
        self.model.write_bytes(struct.pack('<Q', len(raw)) + raw + data)

    def test_strict_json_all_levels(self):
        for raw in ('{"x":1,"x":2}', '{"x":{"y":1,"y":2}}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                m.strict_json(raw)

    def test_deep_json_is_a_structured_input_error(self):
        deep = '{"x":' * 2000 + '0' + '}' * 2000
        # Decoder depth limits vary by Python build; verify exception translation.
        with mock.patch.object(m.json, 'loads', side_effect=RecursionError), self.assertRaises(ValueError):
            m.strict_json(deep)
        p = self.d / 'deep.json'
        p.write_text(deep)
        result = self.cli('--diff', p, p, '--json')
        self.assertEqual(result.returncode, 2)
        self.assertTrue(any(x['severity'] == 'CRIT' for x in json.loads(result.stdout)))
        self.assertNotIn('Traceback', result.stderr)

    def test_incomplete_or_failed_selected_index_blocks_baseline(self):
        p = self.d / 'model.safetensors.index.json'
        for index in ([], {'weight_map': {'absent': 'model.safetensors'}},
                      {'weight_map': {'weight': 'model.safetensors'}, 'metadata': {'total_size': 9}}):
            with self.subTest(index=index):
                p.write_text(json.dumps(index))
                rep = self.audit(do_hashes=True)
                self.assertEqual(rep.exit_code(), 2)
                self.assertFalse(self.model.with_suffix('.safetensors.tensorhashes.json').exists())

    def test_invalid_tensor_schema_blocks_baseline(self):
        for field, value in [('dtype', []), ('shape', None), ('shape', [-1]), ('shape', [True]), ('shape', [1.0]), ('data_offsets', [False, 4]), ('data_offsets', [0, '4'])]:
            with self.subTest(field=field, value=value):
                entry = dict(dtype='F32', shape=[1], data_offsets=[0, 4]); entry[field] = value
                self.raw_header(json.dumps({'weight': entry}), b'\0'*4)
                rep = self.audit(do_hashes=True)
                self.assertEqual(rep.exit_code(), 2)
                self.assertFalse(self.model.with_suffix('.safetensors.tensorhashes.json').exists())

    def test_valid_scalar_empty_tensor_and_empty_artifact(self):
        for tensors in ([('weight', 'F32', [], b'\0'*4)], [('weight', 'F32', [0], b'')], []):
            with self.subTest(tensors=tensors):
                self.model.write_bytes(build(tensors))
                rep = self.audit()
                self.assertFalse(any(s == 'CRIT' for s, _, _ in rep.items))

    def test_layout_gaps_trailing_unknown_dtype_no_baseline(self):
        for data in (build([('weight', 'F32', [1], b'\0'*4)], trailing=b'x'), build([('weight', 'UNKNOWN', [1], b'\0'*4)])):
            self.model.write_bytes(data)
            rep = self.audit(do_hashes=True)
            self.assertEqual(rep.exit_code(), 2)
            self.assertFalse(self.model.with_suffix('.safetensors.tensorhashes.json').exists())

    def test_failure_does_not_hide_independent_incomplete_structure(self):
        header = {'unknown': {'dtype': 'UNKNOWN', 'shape': [1], 'data_offsets': [0,4]},
                  'bad': {'dtype': 'F32', 'shape': [2], 'data_offsets': [4,8]}}
        self.raw_header(json.dumps(header), b'\0'*8)
        rep = self.audit()
        rubric = m.evidence_rubric(rep)
        self.assertEqual(rubric['domains']['structure']['status'], 'FAIL')
        self.assertEqual(rubric['coverage']['status'], 'INCOMPLETE')

    def test_range_and_widening_reject_bad_inputs(self):
        for fn in (m.sha256_at, lambda p,o,n: m.tensor_fingerprints(p,o,n,'F32')):
            for offset, length in ((-1, 1), (True, 1), (0, -1), (0, 999999)):
                with self.subTest(fn=fn, offset=offset, length=length), self.assertRaises(ValueError):
                    fn(self.model, offset, length)
        for dtype in ('F16', 'BF16', 'F32'):
            with self.assertRaises(ValueError): m._to_f32_bytes(b'x', dtype)
            with self.assertRaises(ValueError): m.tensor_fingerprints(self.model, 0, 1, dtype)

    def test_chunk_carry_matches_independent_fingerprints(self):
        raw = struct.pack('<4H', 0, 0x3f80, 0x4000, 0x8000)
        f = self.d / 'raw'; f.write_bytes(raw)
        for chunk in (3, 5):
            with mock.patch.object(m, 'CHUNK', chunk):
                self.assertEqual(m.tensor_fingerprints(f, 0, len(raw), 'BF16'), (H(raw), H(m._to_f32_bytes(raw, 'BF16'))))

    def test_all_template_sources_scan_attack(self):
        attack = "{{ ''.__class__ }}\u200b"
        variants = [('chat_template.jinja', attack), ('other.jinja2', attack), ('additional_chat_templates/tool_use.jinja', attack),
                    ('tokenizer_config.json', json.dumps({'chat_template': attack})),
                    ('tokenizer_config.json', json.dumps({'chat_template': {'tool_use': attack}})),
                    ('tokenizer_config.json', json.dumps({'chat_template': [BENIGN_TEMPLATE, attack]})),
                    ('tokenizer_config.json', json.dumps({'chat_template': [{'name':'tool_use','template':attack}]})),
                    ('chat_template.json', json.dumps({'chat_template': attack}))]
        for name, text in variants:
            with self.subTest(name=name, text=text):
                target = self.d / name; target.parent.mkdir(exist_ok=True); target.write_text(text)
                rep = self.audit()
                self.assertEqual(rep.exit_code(), 2)
                self.assertEqual(m.evidence_rubric(rep)['domains']['templates']['status'], 'CONCERN')
                target.unlink()
                self.config.write_text(json.dumps({'chat_template': BENIGN_TEMPLATE}))

    def test_named_variants_partial_and_conflicts(self):
        self.config.write_text(json.dumps({'chat_template':[{'name':'default','template':BENIGN_TEMPLATE}, {'name':'tool_use','template':'{{ tools }}'}]}))
        rep = self.audit(); self.assertEqual(m.evidence_rubric(rep)['domains']['templates']['status'], 'PASS')
        self.config.write_text(json.dumps({'chat_template':[12, {'name':'default','template':'{{ "".__class__ }}\u200b'}]}))
        rep = self.audit(); self.assertEqual(rep.exit_code(), 2)
        self.assertEqual(m.evidence_rubric(rep)['domains']['templates']['status'], 'NOT CHECKED')
        self.assertTrue(any(x['state']=='inspected' for x in rep.template_coverage))
        self.config.write_text(json.dumps({'chat_template':BENIGN_TEMPLATE}))
        (self.d/'chat_template.jinja').write_text('{{ messages }}')
        self.assertEqual(m.evidence_rubric(self.audit())['domains']['templates']['status'], 'CONCERN')

    def test_bad_utf8_and_config_independence(self):
        (self.d/'chat_template.jinja').write_bytes(b'\xff')
        self.config.write_text(json.dumps({'auto_map': {'AutoTokenizer':'custom.Tokenizer'}, 'chat_template':BENIGN_TEMPLATE}))
        rep = self.audit()
        self.assertTrue(any('auto_map' in msg for _, _, msg in rep.items))
        self.assertEqual(m.evidence_rubric(rep)['coverage']['status'], 'INCOMPLETE')

    def test_exclusive_baselines_and_explicit_new_reference(self):
        a = self.model.with_suffix('.safetensors.tensorhashes.json')
        self.audit(do_hashes=True); original = a.read_bytes()
        result = self.cli(self.model, '--tensor-hashes', '--json')
        self.assertEqual(result.returncode, 1)
        findings = json.loads(result.stdout)
        self.assertTrue(any(x['severity'] == 'WARN' and 'baseline exists:' in x['message'] for x in findings))
        self.assertTrue(any('structure: PASS' in x['message'] for x in findings))
        self.assertFalse(any('structure: NOT CHECKED' in x['message'] for x in findings))
        self.assertEqual(a.read_bytes(), original)
        b = self.d/'B.json'; self.audit(do_hashes=True, baseline_out=b)
        self.assertEqual(json.loads(a.read_text()), json.loads(b.read_text()))
        symlink = self.d/'link.json'; symlink.symlink_to(a)
        self.assertEqual(self.cli(self.model, '--tensor-hashes','--baseline-out',symlink,'--json').returncode, 1)
        self.assertEqual(a.read_bytes(), original)
        self.assertEqual(self.cli(self.model,'--baseline-out',self.d/'x').returncode, 2)

    def test_existing_baseline_keeps_unrelated_critical_and_later_checks(self):
        destination = self.d / 'existing.json'; destination.write_text('preserved')
        self.config.write_text(json.dumps({'chat_template': '{{ messages }}\u200b'}))
        result = self.cli(self.model, '--tensor-hashes', '--baseline-out', destination, '--json')
        findings = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(any(x['severity'] == 'CRIT' and x['section'] == 'template' for x in findings))
        self.assertTrue(any('baseline exists:' in x['message'] for x in findings))
        self.assertTrue(any('structure: PASS' in x['message'] for x in findings))
        self.assertEqual(destination.read_text(), 'preserved')
        with mock.patch.object(m, 'hf_check') as remote:
            self.audit(do_hashes=True, baseline_out=destination, hf_repo='owner/repo')
        remote.assert_called_once()

    def test_only_baseline_exclusive_open_conflict_is_warning(self):
        destination = self.d / 'new.json'
        with mock.patch.object(m.json, 'dump', side_effect=FileExistsError('unrelated failure')):
            with self.assertRaises(FileExistsError):
                self.audit(do_hashes=True, baseline_out=destination)
        self.assertFalse(destination.exists())
        result = self.cli(self.model, '--tensor-hashes', '--baseline-out', self.d/'missing'/'x.json', '--json')
        self.assertEqual(result.returncode, 2)
        self.assertTrue(any('input error:' in x['message'] for x in json.loads(result.stdout)))

    def test_legal_empty_tensor_warning_does_not_block_baseline(self):
        self.model.write_bytes(build([('empty', 'F32', [0], b'')]))
        destination = self.d / 'empty.json'
        rep = self.audit(do_hashes=True, baseline_out=destination)
        self.assertEqual(rep.exit_code(), 1)
        self.assertEqual(m.evidence_rubric(rep)['domains']['structure']['status'], 'PASS')
        self.assertTrue(destination.exists())
        self.assertTrue(any('inventory heuristic' in msg for sev, _, msg in rep.items if sev == 'WARN'))

    def test_baseline_partial_write_cleanup(self):
        destination = self.d/'new.json'
        with mock.patch.object(m.json, 'dump', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.audit(do_hashes=True, baseline_out=destination)
        self.assertFalse(destination.exists())

    def test_json_stdout_and_exit_contract(self):
        result = self.cli(self.model,'--json'); self.assertEqual(result.returncode, 0); self.assertIsInstance(json.loads(result.stdout), list)
        self.config.unlink()
        result = self.cli(self.model,'--json'); self.assertEqual(result.returncode, 0); json.loads(result.stdout)
        for path in (self.d/'missing', self.d):
            result = self.cli(path,'--json'); self.assertEqual(result.returncode, 2); json.loads(result.stdout)
        for raw in ('[]','{"tensors": {"x": "invalid"}}','{"tensors":{},"values":{"unknown":"'+'a'*64+'"}}'):
            bad = self.d/'bad.json'; bad.write_text(raw)
            result = self.cli('--diff',bad,bad,'--json'); self.assertEqual(result.returncode, 2); json.loads(result.stdout)

    def test_rubric_observations_not_findings_order(self):
        rep = self.audit(); before = m.evidence_rubric(rep)
        rep.items.reverse(); self.assertEqual(before, m.evidence_rubric(rep))
        self.assertEqual(before['domains']['provenance']['status'], 'NOT CHECKED')
        self.assertEqual(before['coverage']['status'], 'COMPLETE')
        rep.observe('provenance','NOT CHECKED','lookup failed','HF')
        self.assertEqual(m.evidence_rubric(rep)['coverage']['status'], 'INCOMPLETE')
        rep.observe('provenance','FAIL','mismatch','manifest')
        self.assertEqual(m.evidence_rubric(rep)['domains']['provenance']['status'], 'FAIL')

    def test_ambiguous_manifest_never_pass_or_fail(self):
        sha = H(self.model.read_bytes())
        for hashes in ((sha,sha), (sha,'f'*64)):
            (self.d/'SHA256SUMS').write_text(''.join(f'{h}  {self.model.name}\n' for h in hashes))
            self.assertEqual(m.evidence_rubric(self.audit())['domains']['provenance']['status'], 'CONCERN')

    def test_manifest_whitespace_cannot_alias_another_filename(self):
        digest = m.file_hashes(self.model)[0]
        (self.d / 'SHA256SUMS').write_text(digest + '  other model.safetensors\n')
        rep = self.audit()
        self.assertEqual(m.evidence_rubric(rep)['domains']['provenance']['status'], 'CONCERN')
        self.assertFalse(any(sev == 'OK' and sec == 'manifest' for sev, sec, _ in rep.items))

    def test_metadata_null_config_index_and_duplicate_headers(self):
        for metadata in (None, {"key": 12}):
            self.raw_header(json.dumps({"__metadata__": metadata}))
            self.assertEqual(self.audit().exit_code(), 2)
        self.raw_header('{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{}}', b'\0'*4)
        self.assertEqual(self.audit().exit_code(), 2)
        self.model.write_bytes(build([('weight','F32',[1],b'\0'*4)]))
        for file, obj in [('config.json', []), ('config.json', {'architectures':'bad'}),
                          ('model.safetensors.index.json', {'weight_map':[]} ),
                          ('model.safetensors.index.json', {'weight_map':{'weight': 4}})]:
            target=self.d/file; target.write_text(json.dumps(obj))
            self.assertGreater(self.audit().exit_code(),0); target.unlink()

    def test_tensor_name_cannot_change_coverage(self):
        self.model.write_bytes(build([('not a known type','F32',[1],b'\0'*4)]))
        self.assertEqual(m.evidence_rubric(self.audit())['domains']['structure']['status'],'PASS')

    def test_hf_ambiguity_failure_and_full_revision(self):
        from types import SimpleNamespace as NS
        def sibling(name, digest): return NS(rfilename=name, lfs=NS(sha256=digest))
        sha='a'*64
        cases=[([sibling('model.safetensors',sha)], 'PASS',0),
               ([sibling('a/model.safetensors','b'*64),sibling('b/model.safetensors','c'*64)],'CONCERN',1),
               ([sibling('a',sha),sibling('b',sha)],'CONCERN',1),
               ([sibling('model.safetensors','b'*64)],'FAIL',2), ([], 'NOT CHECKED',1)]
        for siblings,status,code in cases:
            info=NS(sha='d'*40,siblings=siblings)
            module=NS(HfApi=lambda:NS(model_info=lambda *a,**kw:info))
            with mock.patch.dict(sys.modules, {'huggingface_hub':module}):
                rep=m.Report(); m.hf_check('owner/repo',sha,self.model,rep,revision='main')
            self.assertEqual(m.evidence_rubric(rep)['domains']['provenance']['status'],status)
            self.assertEqual(rep.exit_code(),code)
            self.assertTrue(any('d'*40 in text for _,_,text in rep.items))
            self.assertFalse(any('pinned revision: main' in text for _,_,text in rep.items))
        module=NS(HfApi=mock.Mock(side_effect=OSError('offline')))
        with mock.patch.dict(sys.modules, {'huggingface_hub':module}):
            rep=m.Report(); m.hf_check('owner/repo',sha,self.model,rep)
        self.assertEqual(m.evidence_rubric(rep)['coverage']['status'],'INCOMPLETE')

    def test_report_and_diff_subprocess_statuses(self):
        for expected, template in ((0,BENIGN_TEMPLATE),(1,'{{ open }}'),(2,'{{ messages }}\u200b')):
            self.config.write_text(json.dumps({'chat_template':template}))
            result=self.cli(self.model,'-o',self.d/'reports',script='report.py')
            self.assertEqual(result.returncode,expected,result.stderr)
            output=Path(result.stdout.strip()); self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix('.json').exists())
        a,b=self.d/'a.json',self.d/'b.json'
        base={'format':'safetensors','tensors':{'x':'a'*64},'shapes':{'x':[1]}}
        a.write_text(json.dumps(base)); b.write_text(json.dumps(base))
        self.assertEqual(self.cli('--diff',a,b,'--json').returncode,0)
        base['tensors']['x']='b'*64; b.write_text(json.dumps(base))
        result=self.cli('--diff',a,b,'--json'); self.assertEqual(result.returncode,2); json.loads(result.stdout)
        base['format']='gguf'; b.write_text(json.dumps(base))
        result=self.cli('--diff',a,b,'--json'); self.assertEqual(result.returncode,0); json.loads(result.stdout)

    def test_terminal_controls_escaped_json_preserved(self):
        rep=m.Report(); rep.add('WARN','template','payload\x1b[31m\nnext')
        import contextlib,io
        out=io.StringIO()
        with contextlib.redirect_stdout(out): rep.print()
        self.assertNotIn('\x1b',out.getvalue()); self.assertIn('\\x1b',out.getvalue())
        out=io.StringIO()
        with contextlib.redirect_stdout(out): rep.print(as_json=True)
        self.assertEqual(json.loads(out.getvalue())[0]['message'],'payload\x1b[31m\nnext')


class Matching(unittest.TestCase):
    def base(self, tensors, values=None, shapes=None, fmt='safetensors'):
        return dict(format=fmt,tensors=tensors,values=values or {},shapes=shapes or {n:[1] for n in tensors})

    def test_no_double_consumption_both_orders_and_exact_priority(self):
        a = self.base({'a':'x'}, {'a':'y'})
        for tensors in ({'b':'y','c':'x'}, {'c':'x','b':'y'}):
            exact, candidates, ua, ub = m.match_blobs(a,self.base(tensors))
            self.assertEqual(exact,[('a','c')]); self.assertEqual(candidates,[]); self.assertEqual(ub,['b'])
        a = self.base({'a':'x','b':'x'})
        self.assertEqual(len(m.match_blobs(a,self.base({'c':'x','d':'x','e':'x'}))[0]),2)

    def test_subset_and_empty_both_directions(self):
        a, b = self.base({'x':'x','y':'y'}), self.base({'z':'x'})
        self.assertEqual(m.match_blobs(a,b)[2],['y'])
        self.assertEqual(m.match_blobs(b,a)[3],['y'])
        self.assertEqual(m.match_blobs(a,self.base({}))[2],['x','y'])

    def test_same_bytes_different_shape_and_legacy(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.json',Path(d)/'b.json'
            original=self.base({'x':'a'*64},shapes={'x':[2,3]})
            a.write_text(json.dumps(original)); changed=dict(original,shapes={'x':[3,2]}); b.write_text(json.dumps(changed))
            rep=m.Report(); m.diff_baselines(a,b,rep); self.assertEqual(rep.exit_code(),2)
            for variant in (dict(original,shapes={}),dict(original,format='nonsense')):
                b.write_text(json.dumps(variant)); rep=m.Report(); m.diff_baselines(a,b,rep)
                self.assertEqual(m.evidence_rubric(rep)['domains']['baseline']['status'],'CONCERN')
            for path in (a,b): path.write_text(json.dumps(dict(original,format='nonsense')))
            rep=m.Report(); m.diff_baselines(a,b,rep); self.assertEqual(rep.exit_code(),0)


class TypedFingerprints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def baseline(self, name, dtype, raw, fmt, byte_order='little'):
        path = self.d / name; path.write_bytes(raw)
        target = self.d / (name + '.json')
        unit = m.ST_DTYPE_SIZE[dtype]
        tensor = m.STTensor(name, [len(raw) // unit], dtype, 0, len(raw))
        m.write_baseline([tensor], path, H(raw), fmt, m.Report(), target, byte_order)
        return m.read_baseline(target)

    def compare(self, a, b):
        pa, pb = self.d/'A.json', self.d/'B.json'
        pa.write_text(json.dumps(a)); pb.write_text(json.dumps(b))
        rep = m.Report(); m.diff_baselines(pa, pb, rep)
        return rep

    def text(self, rep):
        return '\n'.join(msg for _, _, msg in rep.items)

    def status(self, rep):
        return m.evidence_rubric(rep)['domains']['baseline']['status']

    def pair(self):
        a = self.baseline('source', 'BF16', struct.pack('<H', 0x3f80), 'safetensors')
        b = self.baseline('converted', 'F32', struct.pack('<f', 1.0), 'gguf')
        return a, b

    def test_generated_bf16_f32_pair_has_exact_typed_fingerprints(self):
        a, b = self.pair()
        self.assertEqual(a['values']['source'], H(struct.pack('<I', 0x3f800000)))
        self.assertEqual(b['values']['converted'], b['tensors']['converted'])
        for left, right in ((a,b), (b,a)):
            rep = self.compare(left, right)
            self.assertIn('1 exact normalized-F32 fingerprint matches', self.text(rep))
            self.assertIn('0 normalized-hash candidates', self.text(rep))
            self.assertEqual(rep.exit_code(), 0)
            self.assertEqual(self.status(rep), 'PASS')
            self.assertNotIn('identical values', self.text(rep))
            row = m.evidence_rubric(rep)['domains']['baseline']
            self.assertIn('shapes', row['scope'])
            self.assertIn('NOT VERIFIED', row['scope'])
            self.assertNotIn('shapes', row['reason'])

    def test_known_format_bf16_vs_i32_same_bits_remains_inconclusive(self):
        a, _ = self.pair()
        b = self.baseline('integer', 'I32', struct.pack('<i', 1065353216), 'gguf')
        self.assertEqual(a['values']['source'], b['tensors']['integer'])
        self.assertNotEqual(1.0, 1065353216)
        for left, right in ((a,b), (b,a)):
            rep = self.compare(left, right)
            self.assertIn('0 exact normalized-F32 fingerprint matches', self.text(rep))
            self.assertIn('1 normalized-hash candidates (INCONCLUSIVE)', self.text(rep))
            self.assertEqual(rep.exit_code(), 1)
            self.assertEqual(self.status(rep), 'CONCERN')

    def test_legacy_f32_omission_is_not_type_evidence(self):
        a, b = self.pair()
        for base in (a, b): base.pop('value_metadata')
        b['values'] = {}
        rep = self.compare(a, b)
        self.assertIn('1 normalized-hash candidates (INCONCLUSIVE)', self.text(rep))
        self.assertEqual(self.status(rep), 'CONCERN')

    def test_wrong_or_missing_interpretation_never_promotes_candidate(self):
        a, b = self.pair()
        for key, value in (('dtype','I32'), ('dtype','F16'), ('byte_order','big'),
                           ('normalization','unknown')):
            with self.subTest(key=key, value=value):
                variant = json.loads(json.dumps(b))
                variant['value_metadata']['converted'][key] = value
                rep = self.compare(a, variant)
                self.assertIn('0 exact normalized-F32 fingerprint matches', self.text(rep))
                self.assertEqual(self.status(rep), 'CONCERN')

    def test_malformed_metadata_and_f32_identity_contradictions_rejected(self):
        a, b = self.pair()
        bad = [None, [], {'absent': b['value_metadata']['converted']}, {'converted': {}},
               {'converted': dict(b['value_metadata']['converted'], dtype=32)}]
        for metadata in bad:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.compare(a, dict(b, value_metadata=metadata))
        b['values']['converted'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'must equal its byte hash'):
            self.compare(a, b)

    def test_typed_matches_precede_ambiguous_candidates_and_consume_once(self):
        a, b = self.pair()
        # This untyped normalized hash sorts first, but must not steal the source.
        b['tensors']['aaa-legacy'] = H(b'other raw bytes')
        b['values']['aaa-legacy'] = a['values']['source']
        exact, pairs, only_a, only_b = m.match_blobs(a, b)
        self.assertEqual(exact, [])
        self.assertEqual(pairs, [('source', 'converted')])
        self.assertEqual(only_a, [])
        self.assertEqual(only_b, ['aaa-legacy'])

    def test_exact_matches_keep_priority_over_typed_matches(self):
        a, b = self.pair()
        b['tensors']['raw-copy'] = a['tensors']['source']
        exact, pairs, only_a, only_b = m.match_blobs(a, b)
        self.assertEqual(exact, [('source', 'raw-copy')])
        self.assertEqual(pairs, [])
        self.assertEqual(only_b, ['converted'])

    def test_cross_format_subset_reports_both_sides_and_cannot_pass(self):
        a, b = self.pair()
        a['tensors']['extra-source'] = H(b'extra source')
        rep = self.compare(a, b)
        self.assertIn('not matched in A: extra-source', self.text(rep))
        self.assertEqual(self.status(rep), 'CONCERN')
        rep = self.compare(b, a)
        self.assertIn('not matched in B: extra-source', self.text(rep))
        self.assertEqual(self.status(rep), 'CONCERN')

    def test_empty_inventories_never_pass_for_any_format_pair(self):
        for fa, fb in (('safetensors','gguf'), ('safetensors','safetensors'), (None,None)):
            for ta, tb in (({},{}), ({'a': H(b'a')},{}), ({},{'b': H(b'b')})):
                with self.subTest(formats=(fa,fb), tensors=(ta,tb)):
                    a, b = dict(tensors=ta), dict(tensors=tb)
                    if fa: a['format'] = fa
                    if fb: b['format'] = fb
                    rep = self.compare(a, b)
                    self.assertNotEqual(self.status(rep), 'PASS')
                    self.assertIn('empty tensor', self.text(rep))
                    self.assertGreater(rep.exit_code(), 0)

    def test_cross_format_bytes_do_not_claim_shape_or_dtype_equivalence(self):
        a = dict(format='safetensors', tensors={'a':H(b'bytes')}, shapes={'a':[2,3]})
        b = dict(format='gguf', tensors={'b':H(b'bytes')}, shapes={'b':[3,2]})
        rep = self.compare(a,b)
        self.assertEqual(self.status(rep), 'PASS')
        self.assertIn('1 of 1 B blobs are byte-identical', self.text(rep))
        self.assertNotIn('shapes', m.evidence_rubric(rep)['domains']['baseline']['reason'])

    def test_generic_limits_and_nonmatches_are_info_without_candidate_warning(self):
        a = dict(format='safetensors', tensors={'a':H(b'a')})
        for fmt in ('gguf', 'unknown'):
            rep = self.compare(a, dict(format=fmt, tensors={'b':H(b'b')}))
            self.assertEqual(rep.exit_code(), 0)
            self.assertEqual(self.status(rep), 'CONCERN')
            self.assertTrue(all(sev == 'INFO' for sev, _, _ in rep.items))

    def test_bf16_all_bit_patterns_match_independent_shift_with_chunk_carry(self):
        raw = struct.pack('<65536H', *range(65536))
        expected = b''.join(struct.pack('<I', value << 16) for value in range(65536))
        path = self.d/'bits'; path.write_bytes(raw)
        for chunk in (3, 5, m.CHUNK):
            with self.subTest(chunk=chunk), mock.patch.object(m, 'CHUNK', chunk):
                self.assertEqual(m.tensor_fingerprints(path, 0, len(raw), 'BF16'), (H(raw), H(expected)))

    def test_f16_nan_normalization_has_no_bit_preserving_metadata(self):
        a = self.baseline('f16', 'F16', struct.pack('<H', 0x7c01), 'safetensors')
        self.assertIn('f16', a['values'])
        self.assertNotIn('f16', a['value_metadata'])
        self.assertIsNone(m.normalized_fingerprint(a, 'f16'))

    def test_big_or_unknown_endian_gguf_does_not_record_value_evidence(self):
        for order in ('big', None):
            a = self.baseline(str(order), 'F32', struct.pack('>f', 1.0), 'gguf', order)
            self.assertEqual(a['values'], {})
            self.assertEqual(a['value_metadata'], {})
            self.assertEqual(a['tensors'][str(order)], H(struct.pack('>f', 1.0)))

    @unittest.skipIf(m.gguf is None, 'optional gguf dependency not installed')
    def test_real_gguf_reader_supplies_observed_endian_evidence(self):
        import numpy as np
        for endian, order in ((m.gguf.GGUFEndian.LITTLE, '<'), (m.gguf.GGUFEndian.BIG, '>')):
            with self.subTest(order=order):
                path = self.d / (endian.name + '.gguf')
                writer = m.gguf.GGUFWriter(path, 'clip', endianess=endian)
                writer.add_tensor('weight', np.array([1.0, -0.0], dtype=np.float32))
                writer.write_header_to_file(); writer.write_kv_data_to_file()
                writer.write_tensors_to_file(); writer.close()
                target = self.d / (endian.name + '.json')
                m.audit(path, m.Report(), do_hashes=True, baseline_out=target)
                base = m.read_baseline(target)
                self.assertEqual(base['tensors']['weight'], H(struct.pack(order+'2f', 1.0, -0.0)))
                if endian == m.gguf.GGUFEndian.LITTLE:
                    self.assertEqual(m.normalized_fingerprint(base, 'weight'), H(struct.pack('<2f', 1.0, -0.0)))
                else:
                    self.assertEqual(base['values'], {})
                    self.assertEqual(base['value_metadata'], {})

    @unittest.skipIf(m.gguf is None, 'optional gguf dependency not installed')
    def test_real_safetensors_to_gguf_bf16_f32_diff(self):
        import numpy as np
        source = self.d / 'source.safetensors'
        source.write_bytes(build([('norm', 'BF16', [2], struct.pack('<2H', 0x3f80, 0x4000))]))
        target = self.d / 'target.gguf'
        writer = m.gguf.GGUFWriter(target, 'clip')
        writer.add_tensor('converted_norm', np.array([1.0, 2.0], dtype=np.float32))
        writer.write_header_to_file(); writer.write_kv_data_to_file()
        writer.write_tensors_to_file(); writer.close()
        a, b = self.d/'source.json', self.d/'target.json'
        m.audit(source, m.Report(), do_hashes=True, baseline_out=a)
        m.audit(target, m.Report(), do_hashes=True, baseline_out=b)
        rep = self.compare(m.read_baseline(a), m.read_baseline(b))
        self.assertIn('1 exact normalized-F32 fingerprint matches', self.text(rep))
        self.assertEqual(self.status(rep), 'PASS')
        self.assertEqual(rep.exit_code(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
