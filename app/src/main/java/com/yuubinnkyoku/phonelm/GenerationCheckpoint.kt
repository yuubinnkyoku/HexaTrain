package com.yuubinnkyoku.phonelm

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class GenerationCheckpointCompatibility { COMPATIBLE, INCOMPATIBLE, INVALID }

data class GenerationCheckpoint(
    val path: String,
    val step: Int,
    val seed: Long,
    val tokens: Int,
    val dimension: Int,
    val feedForwardDimension: Int,
    val layers: Int,
    val heads: Int,
    val finite: Boolean,
    val parameterHash: String?,
    val modifiedAtMs: Long,
    val fileSizeBytes: Long,
    val compatibility: GenerationCheckpointCompatibility,
    val formatValid: Boolean,
    val vocabulary: Int = 256,
    val tokenizerKind: String = "byte",
    val tokenizerHash: String? = null,
    val format: String = "NPRTCKPTV2",
    val diagnostic: String? = null,
) {
    val usable: Boolean
        get() = formatValid && compatibility == GenerationCheckpointCompatibility.COMPATIBLE &&
            finite && parameterHash?.matches(Regex("fnv1a64:[0-9a-f]{16}")) == true

    val label: String
        get() = when {
            !formatValid -> "invalid checkpoint"
            compatibility == GenerationCheckpointCompatibility.INCOMPATIBLE ->
                "step $step · D$dimension/FFN$feedForwardDimension · incompatible"
            !finite -> "step $step · D$dimension/FFN$feedForwardDimension · non-finite"
            else -> "step $step · D$dimension/FFN$feedForwardDimension · finite"
        }
}

interface GenerationCheckpointRepository {
    fun listCheckpoints(): List<GenerationCheckpoint>
    fun defaultCheckpoint(checkpoints: List<GenerationCheckpoint> = listCheckpoints()): GenerationCheckpoint?
    fun validateCheckpoint(checkpoint: GenerationCheckpoint): Result<GenerationCheckpoint>
}

internal object GenerationCheckpointSelection {
    fun sorted(checkpoints: List<GenerationCheckpoint>): List<GenerationCheckpoint> =
        checkpoints.sortedWith(
            compareByDescending<GenerationCheckpoint> { it.usable }
                .thenByDescending { it.step }
                .thenByDescending { it.modifiedAtMs }
                .thenBy { it.path },
        )

    fun default(checkpoints: List<GenerationCheckpoint>): GenerationCheckpoint? =
        checkpoints.asSequence().filter { it.usable }
            .sortedWith(compareByDescending<GenerationCheckpoint> { it.step }
                .thenByDescending { it.modifiedAtMs }.thenBy { it.path })
            .firstOrNull()
}

class AppPrivateGenerationCheckpointRepository(
    private val checkpointStore: TrainingCheckpointStore,
    private val expected: TrainingModelConfig = TrainingModelConfig.NICOPEDIA_L19,
) : GenerationCheckpointRepository {
    override fun listCheckpoints(): List<GenerationCheckpoint> = GenerationCheckpointSelection.sorted(
        checkpointStore.listNativeCheckpointPaths().distinct().map { path ->
            GenerationCheckpointInspector.inspect(File(path), expected)
        },
    )

    override fun defaultCheckpoint(checkpoints: List<GenerationCheckpoint>): GenerationCheckpoint? =
        GenerationCheckpointSelection.default(checkpoints)

    override fun validateCheckpoint(checkpoint: GenerationCheckpoint): Result<GenerationCheckpoint> = runCatching {
        val inspected = GenerationCheckpointInspector.inspect(File(checkpoint.path), expected)
        require(inspected.usable) { inspected.diagnostic ?: "checkpoint is not usable" }
        require(inspected.path == checkpoint.path && inspected.step == checkpoint.step &&
            inspected.parameterHash == checkpoint.parameterHash) {
            "checkpoint changed after selection"
        }
        inspected
    }
}

internal object GenerationCheckpointInspector {
    private const val MAGIC_V2 = "NPRTCKPTV2\n"
    private const val MAGIC_V3 = "NPRTCKPTV3\n"
    private const val MAX_FILE_BYTES = 64L * 1024L * 1024L
    private const val MAX_ELEMENTS = 100_000_000L
    private const val FNV_OFFSET = -3750763034362895579L
    private const val FNV_PRIME = 1099511628211L

    fun inspect(file: File, expected: TrainingModelConfig): GenerationCheckpoint {
        val canonical = runCatching { file.canonicalFile }.getOrElse { file.absoluteFile }
        val modified = canonical.takeIf { it.exists() }?.lastModified() ?: 0L
        val size = canonical.takeIf { it.exists() }?.length() ?: 0L
        return runCatching {
            require(canonical.isFile) { "checkpoint file is missing" }
            require(size in 1..MAX_FILE_BYTES) { "checkpoint file size is invalid" }
            canonical.inputStream().use { inspect(it, canonical.path, modified, size, expected) }
        }.getOrElse { error ->
            GenerationCheckpoint(
                path = canonical.path,
                step = 0,
                seed = 0,
                tokens = 0,
                dimension = 0,
                feedForwardDimension = 0,
                layers = 0,
                heads = 0,
                finite = false,
                parameterHash = null,
                modifiedAtMs = modified,
                fileSizeBytes = size,
                compatibility = GenerationCheckpointCompatibility.INVALID,
                formatValid = false,
                diagnostic = error.message ?: "checkpoint inspection failed",
            )
        }
    }

    internal fun inspect(
        source: InputStream,
        path: String,
        modifiedAtMs: Long,
        fileSizeBytes: Long,
        expected: TrainingModelConfig,
    ): GenerationCheckpoint {
        DataInputStream(BufferedInputStream(source, 64 * 1024)).use { input ->
            val magic = ByteArray(MAGIC_V2.length)
            input.readFully(magic)
            val magicText = String(magic, Charsets.US_ASCII)
            require(magicText == MAGIC_V2 || magicText == MAGIC_V3) { "invalid checkpoint magic" }
            val v3 = magicText == MAGIC_V3
            val vocabulary = readPositiveU32(input, "vocabulary")
            val tokens = readPositiveU32(input, "tokens")
            val dimension = readPositiveU32(input, "dimension")
            val feedForward = readPositiveU32(input, "feedForwardDimension")
            val layers = readPositiveU32(input, "layers")
            val heads = readPositiveU32(input, "heads")
            val seed = readPositiveU32(input, "seed")
            val step = readPositiveU32(input, "step")
            require(step < 1_000_000) { "checkpoint step is invalid" }
            val tokenizerKind: String
            val tokenizerHash: String?
            if (v3) {
                tokenizerKind = readBoundedAscii(input, "tokenizer kind", 1..32)
                tokenizerHash = readBoundedAscii(input, "tokenizer hash", 71..71)
                require(tokenizerKind == "byte_bpe" &&
                    tokenizerHash.matches(Regex("sha256:[0-9a-f]{64}"))) {
                    "checkpoint tokenizer identity is invalid"
                }
            } else {
                tokenizerKind = "byte"
                tokenizerHash = null
            }
            require((vocabulary == 1024) == v3) {
                "V1024 requires NPRTCKPTV3 and V256 requires NPRTCKPTV2"
            }
            val namesAndCounts = expectedRegistry(vocabulary, dimension, feedForward, layers)
            var finite = true
            var hash = FNV_OFFSET
            hash = readRegistry(input, namesAndCounts, hashValues = true, initialHash = hash) { finite = finite && it }
            readRegistry(input, namesAndCounts, hashValues = false, initialHash = hash) { finite = finite && it }
            readRegistry(input, namesAndCounts, hashValues = false, initialHash = hash) { finite = finite && it }
            require(input.read() < 0) { "checkpoint has trailing bytes" }
            val compatible = vocabulary == expected.vocabularySize && tokens == expected.tokens &&
                dimension == expected.dimension && feedForward == expected.feedForwardDimension &&
                layers == expected.layers && heads == expected.heads && seed.toLong() == expected.seed &&
                tokenizerKind == expected.tokenizerId.removePrefix("nicopedia-").removeSuffix("-v1") &&
                tokenizerHash == expected.tokenizerHash
            return GenerationCheckpoint(
                path = path,
                step = step,
                seed = seed.toLong(),
                tokens = tokens,
                dimension = dimension,
                feedForwardDimension = feedForward,
                layers = layers,
                heads = heads,
                finite = finite,
                parameterHash = "fnv1a64:" + java.lang.Long.toUnsignedString(hash, 16).padStart(16, '0'),
                modifiedAtMs = modifiedAtMs,
                fileSizeBytes = fileSizeBytes,
                compatibility = if (compatible) GenerationCheckpointCompatibility.COMPATIBLE
                    else GenerationCheckpointCompatibility.INCOMPATIBLE,
                formatValid = true,
                vocabulary = vocabulary,
                tokenizerKind = tokenizerKind,
                tokenizerHash = tokenizerHash,
                format = if (v3) "NPRTCKPTV3" else "NPRTCKPTV2",
            )
        }
    }

    private fun readRegistry(
        input: DataInputStream,
        expected: List<Pair<String, Long>>,
        hashValues: Boolean,
        initialHash: Long,
        onFinite: (Boolean) -> Unit,
    ): Long {
        require(input.readInt() == expected.size) { "checkpoint registry count differs" }
        var hash = initialHash
        expected.forEach { (expectedName, expectedCount) ->
            val nameLength = input.readInt()
            require(nameLength in 1..256) { "checkpoint registry name length is invalid" }
            val nameBytes = ByteArray(nameLength)
            input.readFully(nameBytes)
            val name = String(nameBytes, Charsets.UTF_8)
            val count = input.readLong()
            require(name == expectedName && count == expectedCount && count in 1..MAX_ELEMENTS) {
                "checkpoint registry identity differs"
            }
            if (hashValues) {
                hash = fnv(hash, nameBytes)
                hash = fnv(hash, ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(count).array())
            }
            var remaining = Math.multiplyExact(count, 4L)
            val buffer = ByteArray(64 * 1024)
            var registryFinite = true
            while (remaining > 0) {
                val readSize = minOf(buffer.size.toLong(), remaining).toInt()
                input.readFully(buffer, 0, readSize)
                if (hashValues) hash = fnv(hash, buffer, readSize)
                val floats = ByteBuffer.wrap(buffer, 0, readSize).order(ByteOrder.LITTLE_ENDIAN)
                while (floats.remaining() >= 4) {
                    val value = floats.float
                    registryFinite = registryFinite && value.isFinite()
                }
                remaining -= readSize
            }
            onFinite(registryFinite)
        }
        return hash
    }

    private fun expectedRegistry(vocabulary: Int, dimension: Int, feedForward: Int, layers: Int): List<Pair<String, Long>> {
        require(vocabulary in 2..65_536 && dimension in 1..4_096 && feedForward in 1..16_384 && layers in 1..128) {
            "checkpoint dimensions are outside supported limits"
        }
        val result = mutableListOf("token_embedding" to vocabulary.toLong() * dimension)
        repeat(layers) { layer ->
            val prefix = "layer_${layer.toString().padStart(3, '0')}."
            result += prefix + "norm1_gamma" to dimension.toLong()
            result += prefix + "norm1_beta" to dimension.toLong()
            result += prefix + "wq" to dimension.toLong() * dimension
            result += prefix + "wk" to dimension.toLong() * dimension
            result += prefix + "wv" to dimension.toLong() * dimension
            result += prefix + "wo" to dimension.toLong() * dimension
            result += prefix + "norm2_gamma" to dimension.toLong()
            result += prefix + "norm2_beta" to dimension.toLong()
            result += prefix + "ffn_w1" to dimension.toLong() * feedForward
            result += prefix + "ffn_w2" to feedForward.toLong() * dimension
        }
        result += "output_projection" to dimension.toLong() * vocabulary
        return result
    }

    private fun readPositiveU32(input: DataInputStream, name: String): Int {
        val value = input.readInt()
        require(value > 0) { "checkpoint $name is invalid" }
        return value
    }

    private fun readBoundedAscii(
        input: DataInputStream,
        name: String,
        lengthRange: IntRange,
    ): String {
        val length = input.readInt()
        require(length in lengthRange) { "checkpoint $name length is invalid" }
        val bytes = ByteArray(length)
        input.readFully(bytes)
        return String(bytes, Charsets.US_ASCII)
    }

    private fun fnv(initial: Long, bytes: ByteArray, length: Int = bytes.size): Long {
        var hash = initial
        repeat(length) { index -> hash = (hash xor (bytes[index].toLong() and 0xffL)) * FNV_PRIME }
        return hash
    }
}
