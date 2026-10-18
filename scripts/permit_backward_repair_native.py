"""Exclusive staged permits derived from measured complete gates, never waivers."""
import argparse
import json
from assess_backward_repair import assess,read,require,ROOT
from prepare_saved_p_backward import sha


def make(report,stage):
    build=read(ROOT/'build/fp32-saved-p-backward-repair-final/build-identity.json')
    results={}
    required=['pilot-softmax','pilot-dvalue']
    if stage in ('singles','dual'):required+=['pilot-native-softmax','pilot-native-dvalue']
    if stage=='dual':required+=['pilot-dual']
    for name in required:
        result=assess(report/name)
        require(result['classification']=='PASS' and result['probe_sha256']==build['probe_sha256'],'incomplete or wrong-binary gate '+name)
        require(not result['observer_mismatch_words'] and not result['arithmetic_word_mismatches'],'observer/word entry')
        results[name]=dict(digest=result['numerical_digest'],raw_sha256=sha(report/name/'saved-p-words.bin'),classification=result['classification'])
    forward=[json.loads(l) for l in (report/'fresh-forward-p-joint.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.startswith('{')]
    summaries=[r for r in forward if r.get('kind')=='assessment_summary']
    require(len(summaries)==1 and summaries[0]['joint_pass'] and summaries[0]['required_failed_predicates']==0 and summaries[0]['variant_word_mismatches']==0,'Forward old joint gate')
    fi=read(report/'regressions/double_qk/identity-private.json');base=read(ROOT/'build/fp32-double-qk/build-identity.json')
    require(fi['build']['binary_hashes']==base['binary_hashes'] and fi['exit_code']==0 and fi['focus_takeover_count']==0,'Forward fresh binary/process')
    oldplan=read(ROOT/'host_tests/fp32_custom/saved_p_backward_device_plan.json')
    require(sha(ROOT/'app/src/main/cpp/qnn/qnn_runtime_transformer_training_generalized.inc')==oldplan['production_source_sha256'],'production changed')
    return dict(schema=1,stage=stage,probe_sha256=build['probe_sha256'],dsp_sha256=build['binary_hashes']['libHexaBackwardRepairDsp.so'],
                standalone_pass=True,four_observers_pass=True,forward_joint_pass=True,singles_pass=stage in ('singles','dual'),dual_pass=stage=='dual',
                measured_gates=results,forward_joint_sha256=sha(report/'fresh-forward-p-joint.jsonl'),training='NO-GO')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=__import__('pathlib').Path,required=True);p.add_argument('--stage',choices=('standalone','singles','dual'),required=True);a=p.parse_args()
    result=make(a.report_root,a.stage);out=a.report_root/'permits'/a.stage;out.mkdir(parents=True,exist_ok=False)
    with (out/'native-permit-private.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('backward_repair_measured_permit=PASS stage='+a.stage+' gate_waivers=0')
