package com.yuubinnkyoku.phonelm

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NativeRunArbiterTest {
    @Test
    fun secondNativeOwnerIsRejectedAndCannotReleaseFirst() {
        val first = "test-first"
        val second = "test-second"
        assertTrue(NativeRunArbiter.tryAcquire(first))
        try {
            assertFalse(NativeRunArbiter.tryAcquire(second))
            NativeRunArbiter.release(second)
            assertFalse(NativeRunArbiter.tryAcquire(second))
        } finally {
            NativeRunArbiter.release(first)
        }
        assertTrue(NativeRunArbiter.tryAcquire(second))
        NativeRunArbiter.release(second)
    }
}
