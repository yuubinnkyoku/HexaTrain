"""Separate typed timing/numerical records from actual QNN API/runtime logs."""
import re
import json


def runtime_error_text(stdout,stderr):
    # Keep every actual `api,NAME,RC` and every unstructured runtime diagnostic.
    # Timer records and JSON tensor/numerical/performance records are not API RCs.
    kept=[]
    for line in stdout.splitlines():
        if line.startswith('api_us,'):continue
        if line.startswith('{'):
            try:record=json.loads(line)
            except (ValueError,TypeError):record=None
            if isinstance(record,dict) and record.get('kind') in ('repair_pair','repair_forward','shared_performance','standard_performance'):continue
        kept.append(line)
    return '\n'.join(kept)+'\n'+stderr


def selftest():
    timer='api_us,graphExecute,6031.667\napi_us,graphExecute,6033.177\n'
    numerical='{"kind":"repair_pair","output_hash":"a6031b","median_us":6033.1}\n'
    scoped=runtime_error_text(timer+numerical+'api,graphExecute,0\n','')
    assert not re.search(r'(?<!\d)(6031|6033)(?!\d)',scoped)
    for code in ('6031','6033'):
        for out,err in [('api,graphFinalize,'+code+'\n',''),('','QNN ERROR status '+code),('DMA ERROR '+code+'\n',''),('{"kind":"qnn_error","code":'+code+'}\n','')]:
            assert re.search(r'(?<!\d)'+code+r'(?!\d)',runtime_error_text(timer+out,err))
    print('runtime_error_scope_selftest=PASS timer_numeric_controls=3 actual_error_controls=8 device_calls=0')


if __name__=='__main__':selftest()
