"""Local fixed-SDK API audit. Snippets/absolute paths remain private."""
import argparse
import json
import pathlib
from prepare_saved_p_backward import sha,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--sdk',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    require=a.sdk.resolve()==pathlib.Path('C:/Qualcomm/AIStack/QAIRT/2.48.40.260702').resolve()
    if not require:raise ValueError('fixed SDK only')
    needles={
        'include/QNN/QnnTypes.h':['QNN_PRECISION_FLOAT32','QNN_PRECISION_FLOAT16'],
        'include/QNN/HTP/QnnHtpGraph.h':['QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION','precisionCompensation','ADVANCED_ACTIVATION_FUSION','FINALIZE_OPTIMIZATION_FLAG'],
        'include/QNN/HTP/core/optimize_defs.h':['relaxed_precision_flag','MAKE_OP_FP16_AND_INSERT_CAST'],
        'include/QNN/HTP/core/defopt2qhpi.h':['relaxed_precision_flag','QHPI_FLOAT32'],
        'include/QNN/QnnBackend.h':['QNN_BACKEND_CONFIG_OPTION_CUSTOM']}
    found={}
    for rel,terms in needles.items():
        path=a.sdk/rel;lines=path.read_text(encoding='utf-8').splitlines();matches=[]
        for term in terms:
            selected=[dict(line=i+1,text=l) for i,l in enumerate(lines) if term in l]
            if not selected:raise ValueError('API/implementation audit term absent: '+term)
            matches.append(dict(term=term,matches=selected))
        found[rel]=dict(sha256=sha(path),matches=matches)
    production=ROOT/'app/src/main/cpp/qnn/qnn_runtime_qairt.cpp'
    report=dict(sdk_private=str(a.sdk),files=found,production_source_sha256=sha(production),
        observed_controls=['default null graph config','FLOAT32','FLOAT32 plus precision compensation true'],
        per_op_precision='No public per-standard-op FP32-preservation API identified in audited fixed headers; tensor dtype is distinct from internal arithmetic.',
        numeric_modes='Public precision enums and compensation exist. Internal relaxed_precision_flag rewrite helpers are implementation evidence, not an allowed public configuration knob.',
        optimization='Documented graph optimization options are audited but not swept. No undocumented flags or production modifications.',
        interpretation='API existence does not prove operator precision. The actual fixed-backend diagnostic comparison is separate evidence.')
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print('backward_precision_sdk_audit=PASS documented_controls=3 production_changed=0 undocumented_flags=0')


if __name__=='__main__':main()
