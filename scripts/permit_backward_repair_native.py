"""Exclusive staged permits derived from measured complete gates, never waivers."""
import argparse
import json
import re
from assess_backward_repair import assess,read,require,ROOT
from prepare_saved_p_backward import sha


def pilot_selection(suffix='',overrides=()):
    require(bool(re.fullmatch('[a-z0-9-]{0,32}',suffix)),'unsafe pilot suffix')
    bases=dict(standalone_softmax='pilot-softmax',standalone_dvalue='pilot-dvalue',native_softmax='pilot-native-softmax',native_dvalue='pilot-native-dvalue',dual='pilot-dual')
    folders={mode:name+('-'+suffix if suffix else '') for mode,name in bases.items()}
    seen=set()
    for item in overrides:
        require(isinstance(item,str) and item.count('=')==1,'pilot selection syntax')
        mode,folder=item.split('=')
        require(mode in bases and mode not in seen,'unknown/duplicate pilot role')
        require(bool(re.fullmatch('pilot-[a-z0-9-]{1,58}',folder)),'unsafe pilot directory')
        seen.add(mode);folders[mode]=folder
    return folders


def private_report_path(value):
    path=(ROOT/value).resolve()
    require(path.is_relative_to((ROOT/'build/reports').resolve()),'evidence outside private reports')
    return path


def make(report,stage,suffix='',pilot_directories=(),forward_evidence='initial',pilot_roots=(),forward_control=None):
    folders=pilot_selection(suffix,pilot_directories)
    roots={mode:report for mode in folders}
    selected=set()
    for item in pilot_roots:
        require(item.count('=')==1,'pilot root syntax')
        mode,path=item.split('=')
        require(mode in roots and mode not in selected,'unknown/duplicate pilot root')
        selected.add(mode);roots[mode]=private_report_path(path)
    forward_paths={'initial':('regressions/double_qk','fresh-forward-p-joint.jsonl'),'post-recovery':('post-recovery-forward','post-recovery-forward-joint.jsonl')}
    require(forward_evidence in forward_paths,'unknown Forward evidence')
    forward_folder,forward_joint=forward_paths[forward_evidence]
    build=read(ROOT/'build/fp32-saved-p-backward-repair-final/build-identity.json')
    results={}
    required=['standalone_softmax','standalone_dvalue']
    if stage in ('singles','dual'):required+=['native_softmax','native_dvalue']
    if stage=='dual':required+=['dual']
    for mode in required:
        name=folders[mode]
        directory=roots[mode]/name
        result=assess(directory)
        require(result['mode']==mode and result['classification']=='PASS' and result['probe_sha256']==build['probe_sha256'],'incomplete, wrong-role or wrong-binary gate '+name)
        require(not result['observer_mismatch_words'] and not result['arithmetic_word_mismatches'],'observer/word entry')
        results[name]=dict(digest=result['numerical_digest'],raw_sha256=sha(directory/'saved-p-words.bin'),classification=result['classification'])
    if forward_control:
        from assess_backward_forward import assess as assess_forward
        fresh=private_report_path(forward_control)
        require(assess_forward(fresh)['status']=='PASS','fresh Forward raw joint gate')
        forward_joint=fresh/'forward-joint.jsonl'
        forward_folder=fresh
    forward=[json.loads(l) for l in (report/forward_joint).read_text(encoding='utf-8-sig').splitlines() if l.startswith('{')]
    summaries=[r for r in forward if r.get('kind')=='assessment_summary']
    require(len(summaries)==1 and summaries[0]['joint_pass'] and summaries[0]['required_failed_predicates']==0 and summaries[0]['variant_word_mismatches']==0,'Forward old joint gate')
    fi=read(report/forward_folder/'identity-private.json');base=read(ROOT/'build/fp32-double-qk/build-identity.json')
    require((fi['build']['binary_hashes']==(build['binary_hashes'] if forward_control else base['binary_hashes'])) and fi['exit_code']==0 and fi['focus_takeover_count']==0,'Forward fresh binary/process')
    oldplan=read(ROOT/'host_tests/fp32_custom/saved_p_backward_device_plan.json')
    require(sha(ROOT/'app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc')==oldplan['production_source_sha256'],'production changed')
    return dict(schema=1,stage=stage,pilot_suffix=suffix,pilot_folders=folders,forward_evidence=forward_evidence,probe_sha256=build['probe_sha256'],dsp_sha256=build['binary_hashes']['libHexaBackwardRepairDsp.so'],
                standalone_pass=True,four_observers_pass=True,forward_joint_pass=True,singles_pass=stage in ('singles','dual'),dual_pass=stage=='dual',
                measured_gates=results,forward_joint_sha256=sha(report/forward_joint),pilot_roots_private={mode:str(path) for mode,path in roots.items()},forward_control_private=str(forward_folder) if forward_control else None,training='NO-GO')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=__import__('pathlib').Path,required=True);p.add_argument('--stage',choices=('standalone','singles','dual'),required=True);p.add_argument('--pilot-suffix',default='');p.add_argument('--pilot-directory',action='append',default=[]);p.add_argument('--forward-evidence',choices=('initial','post-recovery'),default='initial');p.add_argument('--pilot-root',action='append',default=[]);p.add_argument('--forward-control');a=p.parse_args()
    result=make(a.report_root,a.stage,a.pilot_suffix,a.pilot_directory,a.forward_evidence,a.pilot_root,a.forward_control);out=a.report_root/'permits'/a.stage;out.mkdir(parents=True,exist_ok=False)
    with (out/'native-permit-private.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('backward_repair_measured_permit=PASS stage='+a.stage+' gate_waivers=0')
