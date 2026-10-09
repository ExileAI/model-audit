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
        self.assertEqual(result.returncode, 2); json.loads(result.stdout)
        self.assertEqual(a.read_bytes(), original)
        b = self.d/'B.json'; self.audit(do_hashes=True, baseline_out=b)
        self.assertEqual(json.loads(a.read_text()), json.loads(b.read_text()))
        symlink = self.d/'link.json'; symlink.symlink_to(a)
        self.assertEqual(self.cli(self.model, '--tensor-hashes','--baseline-out',symlink,'--json').returncode, 2)
        self.assertEqual(a.read_bytes(), original)
        self.assertEqual(self.cli(self.model,'--baseline-out',self.d/'x').returncode, 2)

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
        result=self.cli('--diff',a,b,'--json'); self.assertEqual(result.returncode,1); json.loads(result.stdout)

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
            rep=m.Report(); m.diff_baselines(a,b,rep); self.assertEqual(rep.exit_code(),1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
