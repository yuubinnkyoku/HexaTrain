package com.yuubinnkyoku.phonelm

import java.util.concurrent.atomic.AtomicReference

/**
 * Process-wide ownership for the legacy synchronous JNI bridge.  The native
 * side has one global gRunning flag, so Android benchmark and standalone
 * training must reject a second owner before entering JNI and must never send
 * a stop request to another owner's run.
 */
internal object NativeRunArbiter {
    private val owner = AtomicReference<String?>(null)

    fun tryAcquire(runOwner: String): Boolean = owner.compareAndSet(null, runOwner)

    fun release(runOwner: String) {
        owner.compareAndSet(runOwner, null)
    }

    fun isOwner(runOwner: String): Boolean = owner.get() == runOwner
}
