import copy, importlib.util, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('exporter',Path(__file__).resolve().parents[2]/'scripts/export_public_softmax_projection.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.sha='a'*40
        self.j=dict(variant='oneProjection',exit_code=0,source_sha=self.sha,mode='BACKGROUND_CORRECTNESS',focus_takeover_count=0,protected_identity='PASS',status='COMPLETE',private_sentinel='NEVER_EXPORT',build=dict(source_sha=self.sha,qairt_build_id='2.48.40.260702151143',architecture='v81',fp_contract='off',binary_hashes={k:'b'*64 for k in e.BINARIES}),hashes={**{k:'b'*64 for k in e.BINARIES},**e.FIXTURES})
        lines=['api,contextFree,0','api,deviceFree,0','api,backendFree,0','segmentreceipt,marker=4838310d,raw_inputs=1']
        for mode in ('projection0','projection5'):
            for i in range(30):lines.append(f'onemetric,{mode},{i},qkv0,abs=0.001,rms=0.0001,relative_l2=0.001,max_relative=0.01,fp_abs=0,nonfinite=0,hash=123,limit=0.002,gate=1,cosine=1,fp_word_mismatch=0')
        lines.append('summary,oneProjection,failures=0,cpu_fallback=0,input_mutation=0,finite=1,poison_remaining=0,canary_failures=0,training_updates=0')
        self.s='\n'.join(lines)
    def test_allowlist(self):
        p=e.process(self.j,self.s,'oneProjection',self.sha)
        self.assertEqual(p['required_metric_count'],60);self.assertNotIn('NEVER_EXPORT',str(p))
    def test_fail_closed(self):
        for path,value in [(('exit_code',),3),(('source_sha',),'c'*40),(('focus_takeover_count',),1),(('protected_identity',),'FAIL'),(('build','qairt_build_id'),'2.47'),(('hashes','one_layer_probe'),'d'*64),(('hashes','ln1-parameters.bin'),'e'*64),(('build','fp_contract'),'fast')]:
            with self.subTest(path=path):
                j=copy.deepcopy(self.j);p=j
                for k in path[:-1]:p=p[k]
                p[path[-1]]=value
                with self.assertRaises(ValueError):e.process(j,self.s,'oneProjection',self.sha)
    def test_bad_log(self):
        for s in [self.s.replace('contextFree,0','contextFree,1'),self.s.replace('abs=0.001,','abs=0.01,'),self.s.replace('nonfinite=0','nonfinite=1'),self.s.replace('cpu_fallback=0','cpu_fallback=1'),self.s.replace('api,backendFree,0',''),self.s.replace('projection5','missing'),self.s.replace('abs=0.001,','abs=nan,')]:
            with self.subTest(s=s[-60:]):
                with self.assertRaises(ValueError):e.process(self.j,s,'oneProjection',self.sha)
if __name__=='__main__':unittest.main()
