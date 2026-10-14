"""Negative cohort tests use only private, new build/ copies of supplied records."""
import copy, importlib.util, json, shutil, sys, uuid
from pathlib import Path
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('exporter',root/'scripts/export_public_attention_ln1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
temp=root/'build/reports/g1-fp32-attention-ln1-frontier'/('export-test-'+uuid.uuid4().hex)
temp.mkdir();cohort={}
for lane,paths in original.items():
 cohort[lane]=[]
 for ordinal,path in enumerate(paths):
  dst=temp/(lane+str(ordinal));dst.mkdir()
  for name in ('identity-private.json','stdout.txt'):shutil.copyfile(Path(path)/name,dst/name)
  cohort[lane].append(str(dst))
manifest=temp/'cohort.json';manifest.write_text(json.dumps(cohort));output=temp/'summary.json'
module.export(manifest,output);baseline=output.read_text();assert 'device_private' not in baseline
target=Path(cohort['oneAttention'][0]);identity=target/'identity-private.json';stdout=target/'stdout.txt'
ledger=json.loads(identity.read_text(encoding='utf-8-sig'));lines=stdout.read_text()
tests=0
def rejects(change_ledger=None,change_stdout=None):
 global tests
 mutated=copy.deepcopy(ledger)
 if change_ledger:change_ledger(mutated)
 identity.write_text(json.dumps(mutated));stdout.write_text(change_stdout(lines) if change_stdout else lines)
 try:module.export(manifest,output)
 except (AssertionError,KeyError,ValueError):tests+=1
 else:raise AssertionError('bad cohort accepted')
 identity.write_text(json.dumps(ledger));stdout.write_text(lines)
rejects(lambda x:x.update(source_sha='0'*40))
rejects(lambda x:x['build'].update(source_sha='0'*40))
rejects(lambda x:x.update(exit_code=4))
rejects(lambda x:x.update(focus_takeover_count=1))
rejects(lambda x:x['hashes'].update(one_layer_probe='0'*64))
rejects(lambda x:x['hashes'].update({'one-layer-inputs.bin':'0'*64}))
rejects(lambda x:x['build'].update(qairt_build_id='2.47'))
rejects(change_stdout=lambda s:s.replace('api,graphExecute,0','api,graphExecute,1',1))
rejects(change_stdout=lambda s:s.replace('gate=1,cosine=','gate=0,cosine=',1))
rejects(change_stdout=lambda s:s.replace('api,graphExecute,0\n','',1))
rejects(change_stdout=lambda s:s.replace('cpu_fallback=0','cpu_fallback=1'))
modified=copy.deepcopy(ledger);modified['device_private']={'sentinel':'DO_NOT_EXPORT_PRIVATE_SENTINEL'}
identity.write_text(json.dumps(modified));module.export(manifest,output);assert 'DO_NOT_EXPORT_PRIVATE_SENTINEL' not in output.read_text()
print(f'attention_public_export=PASS baseline=1 rejection_cases={tests} private_sentinel_excluded=1')
