"""Strict allow-list aggregate; derive certification from complete measured data."""
import argparse
import copy
import json
import math
import pathlib
import re
import subprocess
from assess_backward_repair import assess,assess_cohort,require,read,ROOT
from prepare_saved_p_backward import sha
from export_public_saved_p_backward import verified_regressions

MODES=['standalone_softmax','standalone_dvalue','native_softmax','native_dvalue','dual']


def exact_keys(value,names):
    require(isinstance(value,dict) and set(value)==set(names.split()),'missing/forbidden public field')


def validate(value,probe,protected=True):
    exact_keys(value,'schema base source_commit compiled_source_commit bundle_source_commit branch plan_sha256 binary_sha256 fixture_sha256 standard_baseline precision backend host_candidates repair cohort_processes performance forward regression protected_state classifications training limitations')
    require(value['schema']==1 and value['base']=='49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2' and value['training']=='NO-GO','base/training')
    exact_keys(value['binary_sha256'],'probe DSP Host')
    require(value['binary_sha256']['probe']==probe and bool(re.fullmatch('[0-9a-f]{64}',probe)),'wrong binary')
    require(protected and value['protected_state']=='PASS','protected mismatch')
    exact_keys(value['standard_baseline'],'softmax dvalue')
    for name,entry in value['standard_baseline'].items():
        exact_keys(entry,'status core_maxAbs holdout_maxAbs gate observer_words numerical_digest fresh')
        require(entry['status']=='NUMERICAL_FAIL' and entry['fresh'] is True,'old FAIL overwritten')
        expected=.3044056503066239 if name=='softmax' else .035564650305797585
        require(entry['core_maxAbs']==expected and entry['gate']==(.002 if name=='softmax' else .01),'old gate/result changed')
    exact_keys(value['repair'],'standalone_softmax standalone_dvalue native_softmax native_dvalue dual')
    require(value['cohort_processes'] in (0,3),'missing process')
    for mode,entry in value['repair'].items():
        exact_keys(entry,'status consumers core_pairs_per_variant holdout_pairs_per_variant observer_variants observer_word_mismatches arithmetic_word_mismatches numerical_digest processes')
        require(entry['status'] in ('NOT_EXECUTED','API_FAILURE','PILOT_PASS','PASS','NUMERICAL_OR_WORD_FAIL'),'repair status')
        if entry['status'] not in ('NOT_EXECUTED','API_FAILURE'):
            require((entry['core_pairs_per_variant'],entry['holdout_pairs_per_variant'],entry['observer_variants'])==(2200,256,4),'incomplete coverage/observer downgrade')
            require(len(entry['consumers'])==(2 if mode=='dual' else 1),'missing consumer')
            for c in entry['consumers']:
                exact_keys(c,'consumer gate core_maxAbs holdout_maxAbs local_maxAbs total_maxAbs p_induced_actual_maxAbs worst')
                require(c['consumer'] in (0,1) and c['gate']==(.002 if c['consumer']==0 else .01),'wrong gate')
                require(all(isinstance(c[k],(int,float)) and math.isfinite(c[k]) for k in ('core_maxAbs','holdout_maxAbs','local_maxAbs','total_maxAbs','p_induced_actual_maxAbs')),'missing/nonfinite numerical fields')
                if entry['status'] in ('PILOT_PASS','PASS'):
                    require(max(c['core_maxAbs'],c['holdout_maxAbs'],c['total_maxAbs'])<=c['gate'],'false numerical PASS')
            if entry['status'] in ('PILOT_PASS','PASS'):require(entry['observer_word_mismatches']==entry['arithmetic_word_mismatches']==0,'observer/word false PASS')
        if entry['status']=='PASS':require(entry['processes']==3 and value['cohort_processes']==3,'missing formal process')
    if value['cohort_processes']==3:require(all(x['status']=='PASS' and x['processes']==3 for x in value['repair'].values()),'incomplete formal cohort')
    exact_keys(value['precision'],'models internal_precision stage_local_maxAbs primitive_controls observer_cause')
    require(value['precision']['internal_precision']=='INTERNAL_PRECISION_UNRESOLVED','unsupported internal precision certainty')
    exact_keys(value['backend'],'settings production_changed')
    require(value['backend']['production_changed'] is False,'production modified')
    exact_keys(value['regression'],'status checks missing')
    require(value['regression']['status'] in ('PASS','BLOCKED'),'regression status')
    if value['regression']['status']=='PASS':require(not value['regression']['missing'] and all(s=='PASS' for s in value['regression']['checks'].values()),'regression false PASS')
    if value['host_candidates']:
        exact_keys(value['host_candidates'],'softmax_ordered_fp32 softmax_full_double dvalue_ordered_fp32 dvalue_full_double')
        for candidate in value['host_candidates'].values():exact_keys(candidate,'core_maxAbs holdout_maxAbs core_pairs holdout_pairs gate accuracy_pass')
    metric_fields='words word_matches bitwise_mismatches full_case_matches cases maxAbs rms ulp_bins_0_1_2to4_5to16_17to256_257to65536_above65536 signed_zero_mismatches actual_subnormal model_subnormal model_overflow model_underflow_to_zero nonfinite_actual'
    for stage,models in value['precision']['models'].items():
        require(bool(re.fullmatch('[A-Za-z0-9_]+',stage)),'unsafe model stage')
        for name,stats in models.items():
            require(bool(re.fullmatch('[a-z0-9_]+',name)),'unsafe model ID');exact_keys(stats,metric_fields)
    if value['forward']:exact_keys(value['forward'],'p_maxAbs ds_bound dv_bound joint_pass required_failed_predicates variant_word_mismatches core_row_error holdout_row_error')
    if value['performance'] is not None:
        require(all(e['status'] in ('PILOT_PASS','PASS') for e in value['repair'].values()),'performance before accuracy')
        expected={(kind,mode,o,t) for o in (1,0) for t in (1,0) for kind,mode in [('repair_performance',m) for m in MODES+['forward']]+[('standard_performance',m) for m in MODES[:2]]}
        require(len(value['performance'])==32 and {(r['kind'],r['mode'],r['observed'],r['tap']) for r in value['performance']}==expected,'performance coverage')
        for row in value['performance']:
            exact_keys(row,'kind mode observed tap samples first_us median_us p10_us p90_us finalize_us nodes tensors arithmetic_us receipt_inclusive_us')
            require(row['samples']==31,'performance samples')
            if not row['observed']:require(row['arithmetic_us'] is None and row['receipt_inclusive_us'] is None,'invisible timing claim')
    # Nested numeric model/performance payloads are reconstructed from allow-list
    # fields below. Reject raw tensor/identity/path keys at every nesting level.
    forbidden={'raw_tensor','raw_tensors','device','device_id','endpoint','credentials','apk_hash','apk_sha256','checkpoint','corpus','private_log','absolute_path'}
    def safe(x):
        if isinstance(x,dict):
            require(not (set(x)&forbidden),'forbidden raw/identity field')
            for k,v in x.items():safe(k);safe(v)
        elif isinstance(x,list):
            for v in x:safe(v)
        elif isinstance(x,str):require(not re.search(r'[A-Za-z]:[/\\]|(?:^|\s)/(?:Users|data|home|tmp)/|\b\d{1,3}(?:\.\d{1,3}){3}:\d+',x),'private path/endpoint')
        elif isinstance(x,float):require(math.isfinite(x),'nonfinite public number')
    safe(value)


def compact(result,processes):
    consumers=[]
    for v in ([0,1] if result['mode']=='dual' else [int(result['mode'].endswith('dvalue'))]):
        m=result['metrics'];consumers.append(dict(consumer=v,gate=.01 if v else .002,
            core_maxAbs=m['core_'+str(v)]['maxAbs'],holdout_maxAbs=m['holdout_'+str(v)]['maxAbs'],
            local_maxAbs=m['local_'+str(v)]['maxAbs'],total_maxAbs=m['total_'+str(v)]['maxAbs'],
            p_induced_actual_maxAbs=m['p_induced_actual_'+str(v)]['maxAbs'],worst=m['total_'+str(v)]['case']+[m['total_'+str(v)]['coordinate']]))
    return dict(status=('PASS' if processes==3 else 'PILOT_PASS') if result['classification']=='PASS' else 'NUMERICAL_OR_WORD_FAIL',
        consumers=consumers,core_pairs_per_variant=2200,holdout_pairs_per_variant=256,observer_variants=4,
        observer_word_mismatches=result['observer_mismatch_words'],arithmetic_word_mismatches=result['arithmetic_word_mismatches'],numerical_digest=result['numerical_digest'],processes=processes)


def make(report):
    initial=read(report/'initial-private.json');build=read(ROOT/'build/fp32-saved-p-backward-repair-final/build-identity.json')
    require(initial['base']=='49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2','wrong clone/base')
    for p,h in initial['protected'].items():require(sha(ROOT/p)==h,'protected mismatch')
    for name,bundle in initial['bundles'].items():
        for p,h in bundle['source_hashes'].items():require(sha(ROOT/p)==h,'old protected source changed')
        for p,h in bundle.get('binary_hashes',{}).items():require(sha(ROOT/'build'/name/p)==h,'old protected binary changed')
    baseline={}
    for name,mode,folder in [('softmax','standalone_softmax','baseline-softmax-retry'),('dvalue','standalone_dvalue','baseline-dvalue')]:
        from assess_saved_p_backward import assess as assess_standard
        ident=read(report/folder/'identity-private.json');require(ident['exit_code']==4 and ident['forward_limit']==60 and not ident['focus_takeover_count'],'fresh baseline process')
        require(ident['build']['probe_sha256']==initial['bundles']['fp32-saved-p-backward-final']['probe_sha256'],'wrong baseline binary')
        require(read(report/folder/'execution-before-readback-private.json')['exit_code']==4 and read(report/folder/'preflight/device-preflight-private.json')['status']=='PASS','baseline completion/preflight')
        require(sha(report/folder/'saved-p-words.bin')==ident['raw_evidence_sha256'],'fresh baseline raw SHA')
        result=assess_standard(report/folder/'stdout.txt',mode,60)
        baseline[name]=dict(status=result['classification'],core_maxAbs=result['worst']['core']['maxAbs'],holdout_maxAbs=result['worst']['holdout']['maxAbs'],gate=.002 if name=='softmax' else .01,
            observer_words=result['observer_mismatch_words'],numerical_digest=result['numerical_digest'],fresh=True)
    from analyze_backward_precision import analyze
    attr=analyze()
    from assess_backward_primitives import assess as assess_primitives
    primitives=assess_primitives(report/'primitive-controls-complete')
    model_fields='words word_matches bitwise_mismatches full_case_matches cases maxAbs rms ulp_bins_0_1_2to4_5to16_17to256_257to65536_above65536 signed_zero_mismatches actual_subnormal model_subnormal model_overflow model_underflow_to_zero nonfinite_actual'.split()
    models={}
    local={}
    for observer in ('Observed1Tap1','Observed0Tap1'):
        for stage,entry in attr['attribution'][observer].items():
            models[observer+'_'+stage]={name:{k:m[k] for k in model_fields} for name,m in entry['models'].items()}
            local[observer+'_'+stage]=entry['received_double_maxAbs']
    models['dvalue']={name:{k:m[k] for k in model_fields} for name,m in attr['attribution']['dvalue_models'].items()}
    entries={m:dict(status='NOT_EXECUTED',consumers=[],core_pairs_per_variant=0,holdout_pairs_per_variant=0,observer_variants=0,observer_word_mismatches=None,arithmetic_word_mismatches=None,numerical_digest=None,processes=0) for m in MODES}
    folders=['pilot-softmax','pilot-dvalue','pilot-native-softmax','pilot-native-dvalue','pilot-dual']
    for mode,folder in zip(MODES,folders):
        if (report/folder/'identity-private.json').exists():
            identity=read(report/folder/'identity-private.json')
            if identity['exit_code'] in (0,4):entries[mode]=compact(assess(report/folder),1)
            else:
                require(read(report/folder/'execution-before-readback-private.json')['exit_code']==identity['exit_code'],'runtime failure completion')
                require(sha(report/folder/'saved-p-words.bin')==identity['raw_evidence_sha256'],'runtime raw SHA')
                entries[mode]['status']='API_FAILURE'
    cohort_count=0
    if all((report/f'cohort-{i}/identity-private.json').exists() for i in (1,2,3)):
        cohorts=[assess_cohort(report/f'cohort-{i}') for i in (1,2,3)]
        require(all(c['classification']=='PASS' and c['probe_sha256']==build['probe_sha256'] for c in cohorts),'formal cohort failure/wrong binary')
        for mode in MODES:
            require(len({c['roles'][mode]['numerical_digest'] for c in cohorts})==1,'three process digest mismatch')
            entries[mode]=compact(cohorts[0]['roles'][mode],3)
        cohort_count=3
    joint=[json.loads(l) for l in (report/'fresh-forward-p-joint.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.startswith('{')]
    final=[r for r in joint if r.get('kind')=='assessment_summary'];require(len(final)==1 and final[0]['joint_pass'] and final[0]['required_failed_predicates']==0,'old Forward joint gate')
    forward={k:final[0][k] for k in ('p_maxAbs','ds_bound','dv_bound','joint_pass','required_failed_predicates','variant_word_mismatches')}
    forward['core_row_error']=max(r['row_error'] for r in joint if r.get('kind')=='p_case' and r['set']==0)
    forward['holdout_row_error']=max(r['row_error'] for r in joint if r.get('kind')=='p_case' and r['set']==1)
    required=['fast-closeout.txt','host-profile.txt','qnn-profile.txt','host-closeout/identity-private.json','fresh-c0-correctcwd.txt','headless-regression-private.json']+['regressions/'+v+'/identity-private.json' for v in ('mixed_qkv','selective_ln1','double_qk','oneHvx','oneQkvHvx','oneQkScaleA','oneQkScaleB','oneProbabilityScale','oneProbabilityMask','oneProbabilitySoftmax','oneProbabilityP')]
    missing=[p.replace('/','_') for p in required if not (report/p).exists()]
    regression=dict(status='BLOCKED',checks={},missing=missing)
    if not missing:
        checks=verified_regressions(report,initial);checks.pop('New_native_regression_and_formal_cohort')
        regression=dict(status='PASS',checks=checks,missing=[])
    performance=None
    if (report/'performance/identity-private.json').exists():
        ident=read(report/'performance/identity-private.json');require(ident['exit_code']==0 and ident['build']['probe_sha256']==build['probe_sha256'],'performance process/binary')
        rows=[json.loads(l) for l in (report/'performance/stdout.txt').read_text(encoding='utf-8-sig').splitlines() if l.startswith('{')]
        rows=[r for r in rows if r.get('kind') in ('repair_performance','standard_performance')]
        require(len(rows)==32 and all(r['samples']==31 for r in rows),'performance coverage')
        fields='kind mode observed tap samples first_us median_us p10_us p90_us finalize_us nodes tensors arithmetic_us receipt_inclusive_us'.split()
        performance=[{k:r[k] for k in fields} for r in rows]
    settings=[]
    for name in ('config-fp32-pilot','config-compensation-pilot','config-dvalue-1-resume','config-dvalue-2-resume'):
        if (report/name/'identity-private.json').exists():
            i=read(report/name/'identity-private.json');settings.append(dict(setting=name,process_exit=i['exit_code'],forward_limit=i['forward_limit'],mode=i['mode']))
    classes=['STANDARD_BACKWARD_PRECISION_ATTRIBUTED','STANDARD_BACKWARD_PRECISION_MODEL_SUPPORTED']
    if cohort_count==3:
        classes+=['CUSTOM_SOFTMAX_BACKWARD_STANDALONE_VERIFIED','CUSTOM_DVALUE_STANDALONE_VERIFIED','BACKWARD_OBSERVER_INDEPENDENCE_VERIFIED','SAVED_P_SINGLE_NATIVE_VERIFIED','SAVED_P_DUAL_NATIVE_VERIFIED','SAVED_P_BACKWARD_TOTAL_GRADIENT_VERIFIED']
    else:classes+=['UNRESOLVED']
    if performance is not None:classes+=['BACKWARD_REPAIR_PERFORMANCE_MEASURED']
    if entries['standalone_softmax']['status']=='API_FAILURE':classes+=['BLOCKED_SOFTMAX_BACKWARD_REPAIR']
    if entries['standalone_dvalue']['status']=='API_FAILURE':classes+=['BLOCKED_DVALUE_REPAIR']
    if not any(entries[m]['status']!='NOT_EXECUTED' for m in MODES):classes+=['BLOCKED_DEVICE_SAFETY']
    value=dict(schema=1,base=initial['base'],source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        compiled_source_commit=build['compiled_source_sha'],bundle_source_commit=build['bundle_source_sha'],branch='investigation/g1-backward-primitive-precision-repair',
        plan_sha256=sha(ROOT/'host_tests/fp32_custom/backward_primitive_precision_plan.json'),binary_sha256=dict(probe=build['probe_sha256'],DSP=build['binary_hashes']['libHexaBackwardRepairDsp.so'],Host=build['binary_hashes']['libHexaBackwardRepairHost.so']),
        fixture_sha256=dict(P=sha(ROOT/'build/fp32-double-qk/double-inputs.bin'),gradients=build['gradient_sha256']),standard_baseline=baseline,
        precision=dict(models=models,internal_precision='INTERNAL_PRECISION_UNRESOLVED',stage_local_maxAbs=local,primitive_controls=dict(cases=576,ranks=[2,4],layout_controls_pass=primitives['layout_controls_pass'],nonfinite_output_words=primitives['nonfinite_output_words']),observer_cause='Tap-dependent graph materialization characterized; first invisible NoTap internal boundary and proprietary lowering remain unresolved.'),
        backend=dict(settings=settings,production_changed=False),host_candidates={name:{k:c[k] for k in ('core_maxAbs','holdout_maxAbs','core_pairs','holdout_pairs','gate','accuracy_pass')} for name,c in attr['host_candidates'].items()},repair=entries,cohort_processes=cohort_count,performance=performance,forward=forward,regression=regression,protected_state='PASS',classifications=classes,training='NO-GO',
        limitations=['Fixed diagnostic G1 rank4 DSP geometry and frozen reference-derived gradients; rank2 custom compatibility is Host-only.', 'No full Attention, full layer, or training certification.', 'Old 35/40 Tap incident REPRODUCED / CAUSE UNRESOLVED; legacy 6031 remains unresolved. A new standard FLOAT32 MatMul pilot returned API6031 and process139; causal attribution unresolved.', 'NoReceipt/NoTap input P is not directly observed; its own final accuracy and observer words are checked.'])
    validate(value,build['probe_sha256'])
    return value


def selftest():
    empty=dict(status='NOT_EXECUTED',consumers=[],core_pairs_per_variant=0,holdout_pairs_per_variant=0,observer_variants=0,observer_word_mismatches=None,arithmetic_word_mismatches=None,numerical_digest=None,processes=0)
    value=dict(schema=1,base='49d91b67e1ca4b7b7ede66231e5b7abfe1220ec2',source_commit='a'*40,compiled_source_commit='a'*40,bundle_source_commit='a'*40,branch='investigation/test',plan_sha256='a'*64,binary_sha256=dict(probe='a'*64,DSP='a'*64,Host='a'*64),fixture_sha256={},
        standard_baseline={n:dict(status='NUMERICAL_FAIL',core_maxAbs=.3044056503066239 if n=='softmax' else .035564650305797585,holdout_maxAbs=.01,gate=.002 if n=='softmax' else .01,observer_words=0,numerical_digest='a'*64,fresh=True) for n in ('softmax','dvalue')},precision=dict(models={},internal_precision='INTERNAL_PRECISION_UNRESOLVED',stage_local_maxAbs={},primitive_controls={},observer_cause='unresolved'),backend=dict(settings=[],production_changed=False),host_candidates={},repair={m:copy.deepcopy(empty) for m in MODES},cohort_processes=0,performance=None,forward={},regression=dict(status='BLOCKED',checks={},missing=['device']),protected_state='PASS',classifications=['UNRESOLVED'],training='NO-GO',limitations=[])
    # A compact fixture with one correctly measured full standalone pilot.
    entry=dict(status='PILOT_PASS',consumers=[dict(consumer=0,gate=.002,core_maxAbs=.0001,holdout_maxAbs=.00001,local_maxAbs=.0001,total_maxAbs=.0001,p_induced_actual_maxAbs=0.,worst=[1,1,0,0,0,0,0])],core_pairs_per_variant=2200,holdout_pairs_per_variant=256,observer_variants=4,observer_word_mismatches=0,arithmetic_word_mismatches=0,numerical_digest='a'*64,processes=1)
    value['repair']['standalone_softmax']=entry;validate(value,'a'*64)
    mutations=[lambda x:x['repair']['standalone_softmax']['consumers'][0].update(total_maxAbs=.1),lambda x:x['repair']['standalone_softmax'].update(core_pairs_per_variant=2199),lambda x:x['repair']['standalone_softmax'].update(status='PASS'),lambda x:x['binary_sha256'].update(probe='b'*64),lambda x:x['standard_baseline']['softmax'].update(status='PASS'),lambda x:x['repair']['standalone_softmax'].update(observer_variants=3),lambda x:x['repair']['standalone_softmax']['consumers'][0].update(gate=.01),lambda x:x['repair']['standalone_softmax']['consumers'][0].pop('core_maxAbs'),lambda x:x.update(raw_tensor=[1]),lambda x:x.update(protected_state='FAIL')]
    for mutation in mutations:
        bad=copy.deepcopy(value);mutation(bad)
        try:validate(bad,'a'*64)
        except (ValueError,KeyError):pass
        else:raise AssertionError('unsafe public aggregate accepted')
    print('backward_precision_exporter_selftest=PASS rejections=10 device_calls=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--report-root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
    if a.self_test:selftest()
    else:
        result=make(a.report_root)
        with a.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        print('backward_precision_public_export=PASS cohort_processes='+str(result['cohort_processes']))
