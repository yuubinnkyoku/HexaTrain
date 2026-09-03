package com.yuubinnkyoku.phonelm

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class GenerationCheckpointImportTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test fun validCheckpointPublishesByDirectoryRename() {
        val roots = roots()
        val staged = stage(roots.first, "candidate", legacyCheckpointBytes(1f))
        val expected = identity(staged.resolve("model.ckpt"))

        val result = GenerationCheckpointImport.publish(request("candidate", roots, expected))

        assertFalse(result.idempotent)
        assertTrue(result.publishedDirectory.resolve("model.ckpt").isFile)
        assertFalse(staged.exists())
        assertTrue(result.checkpoint.usable)
        assertEquals(250, result.checkpoint.step)
    }

    @Test fun publishedCheckpointIsVisibleThroughGenerationRepositoryDiscovery() {
        val roots = roots()
        val staged = stage(roots.first, "discoverable", legacyCheckpointBytes(1f))
        val expected = identity(staged.resolve("model.ckpt"))
        val result = GenerationCheckpointImport.publish(request("discoverable", roots, expected))
        val store = object : TrainingCheckpointStore {
            override fun list() = emptyList<TrainingCheckpointMetadata>()
            override fun save(metadata: TrainingCheckpointMetadata) = Unit
            override fun archive(metadata: TrainingCheckpointMetadata) = Unit
            override fun listNativeCheckpointPaths() = listOf(result.publishedDirectory.resolve("model.ckpt").path)
        }

        val listed = AppPrivateGenerationCheckpointRepository(store).listCheckpoints()

        assertEquals(1, listed.size)
        assertTrue(listed.single().usable)
        assertEquals(expected.parameterHash, listed.single().parameterHash)
    }

    @Test fun missingCheckpointFailsAndOnlyRemovesItsStagingDirectory() {
        val roots = roots()
        val staged = File(roots.first, "missing.tmp").apply { mkdirs() }

        val failure = runCatching { GenerationCheckpointImport.publish(request("missing", roots, dummyIdentity())) }

        assertTrue(failure.isFailure)
        assertFalse(staged.exists())
        assertFalse(File(roots.second, "imported/missing").exists())
    }

    @Test fun wrongBpeTokenizerFailsClosed() {
        val roots = roots()
        val staged = stage(roots.first, "wrong-tokenizer", bpeCheckpointBytes())
        staged.resolve("byte-bpe-v1024.model").writeBytes(byteArrayOf(1, 2, 3))
        val expected = identity(staged.resolve("model.ckpt"))

        val failure = runCatching { GenerationCheckpointImport.publish(request("wrong-tokenizer", roots, expected)) }

        assertTrue(failure.isFailure)
        assertFalse(staged.exists())
        assertFalse(File(roots.second, "imported/wrong-tokenizer").exists())
    }

    @Test fun missingBpeTokenizerFailsClosed() {
        val roots = roots()
        val staged = stage(roots.first, "missing-tokenizer", bpeCheckpointBytes())
        val expected = identity(staged.resolve("model.ckpt"))

        val failure = runCatching {
            GenerationCheckpointImport.publish(request("missing-tokenizer", roots, expected))
        }

        assertTrue(failure.isFailure)
        assertFalse(staged.exists())
        assertFalse(File(roots.second, "imported/missing-tokenizer").exists())
    }

    @Test fun unexpectedStagingEntryFailsClosed() {
        val roots = roots()
        val staged = stage(roots.first, "extra", legacyCheckpointBytes(1f))
        staged.resolve("unexpected.bin").writeBytes(byteArrayOf(9))
        val expected = identity(staged.resolve("model.ckpt"))

        val failure = runCatching { GenerationCheckpointImport.publish(request("extra", roots, expected)) }

        assertTrue(failure.isFailure)
        assertFalse(staged.exists())
        assertFalse(File(roots.second, "imported/extra").exists())
    }

    @Test fun nonFiniteCheckpointFailsClosed() {
        val roots = roots()
        val staged = stage(roots.first, "nonfinite", legacyCheckpointBytes(Float.NaN))
        val expected = identity(staged.resolve("model.ckpt"))

        val failure = runCatching { GenerationCheckpointImport.publish(request("nonfinite", roots, expected)) }

        assertTrue(failure.isFailure)
        assertFalse(staged.exists())
        assertFalse(File(roots.second, "imported/nonfinite").exists())
    }

    @Test fun identicalRootsAreRejectedWithoutTouchingStaging() {
        val root = temporary.newFolder("single-root")
        val staged = stage(root, "same-root", legacyCheckpointBytes(1f))
        val expected = identity(staged.resolve("model.ckpt"))

        val failure = runCatching {
            GenerationCheckpointImport.publish(
                GenerationCheckpointImportRequest("same-root", root, root, expected),
            )
        }

        assertTrue(failure.isFailure)
        assertTrue(staged.exists())
    }

    @Test fun headerIdentityMismatchFailsAndDoesNotPublish() {
        val roots = roots()
        val staged = stage(roots.first, "identity", legacyCheckpointBytes(1f))
        val actual = identity(staged.resolve("model.ckpt"))
        val wrongStep = actual.copy(step = actual.step + 1)

        assertTrue(runCatching { GenerationCheckpointImport.publish(request("identity", roots, wrongStep)) }.isFailure)
        assertFalse(staged.exists())
        assertFalse(File(roots.second, "imported/identity").exists())
    }

    @Test fun sameBytesAreIdempotentButSameIdDifferentBytesIsRejected() {
        val roots = roots()
        val first = stage(roots.first, "same", legacyCheckpointBytes(1f))
        val expected = identity(first.resolve("model.ckpt"))
        assertFalse(GenerationCheckpointImport.publish(request("same", roots, expected)).idempotent)

        stage(roots.first, "same", legacyCheckpointBytes(1f))
        assertTrue(GenerationCheckpointImport.publish(request("same", roots, expected)).idempotent)

        val conflicting = stage(roots.first, "same", legacyCheckpointBytes(2f))
        val conflictExpected = identity(conflicting.resolve("model.ckpt"))
        assertTrue(runCatching { GenerationCheckpointImport.publish(request("same", roots, conflictExpected)) }.isFailure)
        assertFalse(conflicting.exists())
        assertTrue(File(roots.second, "imported/same/model.ckpt").isFile)
    }

    @Test fun importIdCannotTraverseRootsAndDoesNotCreateTrainingMetadata() {
        val roots = roots()
        val staged = stage(roots.first, "safe", legacyCheckpointBytes(1f))
        val expected = identity(staged.resolve("model.ckpt"))
        assertTrue(runCatching { GenerationCheckpointImport.publish(request("../escape", roots, expected)) }.isFailure)
        assertTrue(staged.exists()) // Invalid request must not delete an unrelated staging directory.

        val result = GenerationCheckpointImport.publish(request("safe", roots, expected))
        assertTrue(result.publishedDirectory.path.contains("imported${File.separator}safe"))
        // The importer has no TrainingCheckpointStore parameter and never writes resume metadata.
        assertEquals(emptyList<TrainingCheckpointMetadata>(), InMemoryTrainingCheckpointStore().list())
    }

    private fun roots(): Pair<File, File> =
        temporary.newFolder("staging") to temporary.newFolder("production")

    private fun stage(stagingRoot: File, id: String, bytes: ByteArray): File =
        File(stagingRoot, "$id.tmp").apply {
            mkdirs()
            resolve("model.ckpt").writeBytes(bytes)
        }

    private fun request(
        id: String,
        roots: Pair<File, File>,
        expected: GenerationCheckpointImportIdentity,
    ) = GenerationCheckpointImportRequest(id, roots.first, roots.second, expected)

    private fun identity(file: File): GenerationCheckpointImportIdentity {
        val checkpoint = GenerationCheckpointInspector.inspect(file)
        return GenerationCheckpointImportIdentity(
            checkpointSha256 = sha256(file), checkpointSizeBytes = file.length(),
            vocabulary = checkpoint.vocabulary, tokens = checkpoint.tokens, dimension = checkpoint.dimension,
            feedForwardDimension = checkpoint.feedForwardDimension, layers = checkpoint.layers,
            heads = checkpoint.heads, step = checkpoint.step, seed = checkpoint.seed,
            tokenizerKind = checkpoint.tokenizerKind, tokenizerHash = checkpoint.tokenizerHash,
            parameterHash = checkpoint.parameterHash!!,
        )
    }

    private fun dummyIdentity() = GenerationCheckpointImportIdentity(
        "sha256:" + "00".repeat(32), 1, 256, 32, 32, 32, 19, 2, 250, 1,
        "byte", null, "fnv1a64:0000000000000000",
    )

    private fun sha256(file: File): String = file.inputStream().use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(8192)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
        "sha256:" + digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun legacyCheckpointBytes(value: Float): ByteArray = checkpointBytes(
        magic = "NPRTCKPTV2\n", vocabulary = 256, tokenizerHash = null, value = value,
    )

    private fun bpeCheckpointBytes(): ByteArray = checkpointBytes(
        magic = "NPRTCKPTV3\n", vocabulary = 1024,
        tokenizerHash = ModelConfigurationCatalog.CANONICAL_BPE_TOKENIZER_HASH, value = 1f,
    )

    private fun checkpointBytes(magic: String, vocabulary: Int, tokenizerHash: String?, value: Float): ByteArray {
        val dimension = 32
        val feedForward = 32
        val layers = 19
        val registry = buildList {
            add("token_embedding" to vocabulary.toLong() * dimension)
            repeat(layers) { layer ->
                val prefix = "layer_%03d.".format(layer)
                add(prefix + "norm1_gamma" to dimension.toLong()); add(prefix + "norm1_beta" to dimension.toLong())
                listOf("wq", "wk", "wv", "wo").forEach { add(prefix + it to dimension.toLong() * dimension) }
                add(prefix + "norm2_gamma" to dimension.toLong()); add(prefix + "norm2_beta" to dimension.toLong())
                add(prefix + "ffn_w1" to dimension.toLong() * feedForward); add(prefix + "ffn_w2" to feedForward.toLong() * dimension)
            }
            add("output_projection" to dimension.toLong() * vocabulary)
        }
        return ByteArrayOutputStream().also { output ->
            DataOutputStream(output).use { data ->
                data.write(magic.toByteArray(Charsets.US_ASCII))
                listOf(vocabulary, 32, dimension, feedForward, layers, 2, 1, 250).forEach(data::writeInt)
                if (tokenizerHash != null) {
                    val kind = "byte_bpe".toByteArray(Charsets.US_ASCII)
                    data.writeInt(kind.size); data.write(kind)
                    val hash = tokenizerHash.toByteArray(Charsets.US_ASCII)
                    data.writeInt(hash.size); data.write(hash)
                }
                repeat(3) {
                    data.writeInt(registry.size)
                    registry.forEach { (name, count) ->
                        val encoded = name.toByteArray(Charsets.UTF_8)
                        data.writeInt(encoded.size); data.write(encoded); data.writeLong(count)
                        repeat(count.toInt()) { data.writeInt(Integer.reverseBytes(value.toBits())) }
                    }
                }
            }
        }.toByteArray()
    }
}
