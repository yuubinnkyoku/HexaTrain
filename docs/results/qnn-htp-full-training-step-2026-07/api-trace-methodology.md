# QNN HTP API trace evidence methodology

This evidence is an allow-list export from one successful PhoneLM full-step run on the locally installed QAIRT 2.48 runtime. The public file contains no raw run, QNN callback message, absolute path, function pointer address, ADB endpoint, IP address, PC user name, APK, or Qualcomm binary.

`api_trace_*_result` fields are the actual QNN API return codes observed by PhoneLM. Handle fields record whether the output handle was non-null after the corresponding call. The execute attempt, success, failure, first-result, last-result, and first-failure fields come from PhoneLM runtime counters; they are not copied from the requested step count.

The `*_symbol_library` values are shared-object basenames returned by Android `dladdr` for function pointers in the selected QNN provider function table. They are not pointer addresses and do not expose the absolute library path.

QNN callback text is supplied by the Qualcomm runtime to PhoneLM's registered callback. It is bounded and retained only in local evidence-mode raw reports; the allow-list exporter excludes the entire callback block. The published run used counter-only mode with callback capture disabled and QNN log level WARN.

`htp_full_step_used` is a PhoneLM aggregate decision based on successful graph preparation/execution and correctness checks. The trace demonstrates calls through a provider table whose functions resolve to `libQnnHtp.so`; it is not an NPU utilization percentage or a hardware performance counter.