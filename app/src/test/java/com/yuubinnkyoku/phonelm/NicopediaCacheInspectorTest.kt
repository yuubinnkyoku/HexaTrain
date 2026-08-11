package com.yuubinnkyoku.phonelm

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NicopediaCacheInspectorTest {
    @Test
    fun validatesNprtByteV1AndBuildsStableIdentity() {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { output ->
            output.write("NPRTBYTEV1\n".toByteArray(Charsets.US_ASCII))
            output.writeInt(32)
            output.writeInt(256)
            output.writeLong(1L)
            output.writeLong(0x1122334455667788L)
            output.write(ByteArray(33) { it.toByte() })
        }

        val inspection = NicopediaCacheInspector.inspect(
            ByteArrayInputStream(bytes.toByteArray()),
            TrainingModelConfig.NICOPEDIA_L19,
        )

        assertEquals(32, inspection.context)
        assertEquals(256, inspection.vocabulary)
        assertEquals(1L, inspection.recordCount)
        assertTrue(inspection.identity.startsWith("NPRTBYTEV1;context=32;vocab=256;records=1;"))
        assertTrue(inspection.identity.contains("sha256="))
        assertTrue(inspection.identity.contains("fnv1a64:"))
        assertTrue(inspection.identity.contains("training_order=fnv1a64:"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsConfigMismatchBeforeNativeStaging() {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { output ->
            output.write("NPRTBYTEV1\n".toByteArray(Charsets.US_ASCII))
            output.writeInt(16)
            output.writeInt(256)
            output.writeLong(1L)
            output.writeLong(1L)
            output.write(ByteArray(17))
        }

        NicopediaCacheInspector.inspect(
            ByteArrayInputStream(bytes.toByteArray()),
            TrainingModelConfig.NICOPEDIA_L19,
        )
    }
}
