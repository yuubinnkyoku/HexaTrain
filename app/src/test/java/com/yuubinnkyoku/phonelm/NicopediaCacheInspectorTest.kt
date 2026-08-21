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

    @Test fun validatesCanonicalNprtBpeV1AndTokenRange() {
        val config = ModelConfigurationCatalog.config(1024, 64, 64)
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { output ->
            output.write("NPRTBPEV1\n".toByteArray(Charsets.US_ASCII))
            output.writeInt(32)
            output.writeInt(1024)
            output.write(hexToBytes(ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH.removePrefix("sha256:")))
            output.writeLong(1L)
            output.writeLong(0x1122334455667788L)
            repeat(33) { output.writeShort((it * 31) % 1024) }
        }
        val inspection = NicopediaCacheInspector.inspect(ByteArrayInputStream(bytes.toByteArray()), config)
        assertEquals("NPRTBPEV1", inspection.format)
        assertEquals(1024, inspection.vocabulary)
        assertEquals(ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH, inspection.tokenizerHash)
        assertTrue(inspection.identity.contains("tokenizer=byte_bpe"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsBpeCacheWithWrongCanonicalTokenizerHash() {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { output ->
            output.write("NPRTBPEV1\n".toByteArray(Charsets.US_ASCII))
            output.writeInt(32); output.writeInt(1024); output.write(ByteArray(32)); output.writeLong(1L)
            output.writeLong(1L); repeat(33) { output.writeShort(0) }
        }
        NicopediaCacheInspector.inspect(
            ByteArrayInputStream(bytes.toByteArray()),
            ModelConfigurationCatalog.config(1024, 32, 32),
        )
    }

    private fun hexToBytes(value: String): ByteArray = ByteArray(value.length / 2) { index ->
        value.substring(index * 2, index * 2 + 2).toInt(16).toByte()
    }
}
