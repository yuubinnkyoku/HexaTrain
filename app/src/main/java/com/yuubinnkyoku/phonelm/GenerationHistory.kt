package com.yuubinnkyoku.phonelm

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption

enum class GenerationHistoryStatus { SUCCESS, FAILED, CANCELLED }

data class GenerationHistoryRecord(
    val id: String,
    val createdAtMs: Long,
    val promptBytes: ByteArray,
    val mode: GenerationMode,
    val temperature: Float,
    val topK: Int,
    val samplingSeed: Long,
    val maxNewBytes: Int,
    val checkpointStep: Int,
    val checkpointParameterHash: String,
    val vocabulary: Int,
    val tokens: Int,
    val dimension: Int,
    val feedForwardDimension: Int,
    val layers: Int,
    val heads: Int,
    val generatedBytes: ByteArray,
    val elapsedMs: Long,
    val backend: String,
    val qnnExecuteAttempts: Long,
    val qnnExecuteSuccesses: Long,
    val qnnExecuteFailures: Long,
    val cpuFallback: Boolean,
    val finite: Boolean,
    val status: GenerationHistoryStatus,
    val failureMessage: String? = null,
    val checkpointFormat: String = "NPRTCKPTV2",
    val tokenizerKind: String = "byte",
    val tokenizerHash: String? = null,
    val generatedTokenCount: Int = generatedBytes.size,
)

data class GenerationHistoryItem(
    val record: GenerationHistoryRecord,
    val promptText: String,
    val outputText: String,
)

data class GenerationHistoryUiState(
    val items: List<GenerationHistoryItem> = emptyList(),
    val selectedId: String? = null,
    val loading: Boolean = false,
    val message: String? = null,
) {
    val selected: GenerationHistoryItem?
        get() = items.firstOrNull { it.record.id == selectedId }
}

fun interface LosslessByteDisplay {
    fun display(bytes: ByteArray): String
}

interface GenerationHistoryRepository {
    fun insert(record: GenerationHistoryRecord): Boolean
    fun listNewestFirst(): List<GenerationHistoryRecord>
    fun delete(id: String): Boolean
    fun clear()
}

class FileGenerationHistoryRepository(private val file: File) : GenerationHistoryRepository {
    private val lock = Any()

    override fun insert(record: GenerationHistoryRecord): Boolean = synchronized(lock) {
        val records = readAll().toMutableList()
        if (records.any { it.id == record.id }) return@synchronized false
        records += record
        writeAll(records)
        true
    }

    override fun listNewestFirst(): List<GenerationHistoryRecord> = synchronized(lock) {
        readAll().sortedWith(compareByDescending<GenerationHistoryRecord> { it.createdAtMs }.thenByDescending { it.id })
    }

    override fun delete(id: String): Boolean = synchronized(lock) {
        val records = readAll()
        val remaining = records.filterNot { it.id == id }
        if (remaining.size == records.size) return@synchronized false
        writeAll(remaining)
        true
    }

    override fun clear() = synchronized(lock) {
        if (file.exists() && !file.delete()) {
            writeAll(emptyList())
        }
        Unit
    }

    private fun readAll(): List<GenerationHistoryRecord> {
        if (!file.isFile) return emptyList()
        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
            val magic = input.readUTF()
            require(magic == MAGIC_V1 || magic == MAGIC_V2) { "generation history format is invalid" }
            val v2 = magic == MAGIC_V2
            val count = input.readInt()
            require(count in 0..MAX_RECORDS) { "generation history record count is invalid" }
            return List(count) { readRecord(input, v2) }.also {
                require(input.read() < 0) { "generation history has trailing data" }
            }
        }
    }

    private fun writeAll(records: List<GenerationHistoryRecord>) {
        require(records.size <= MAX_RECORDS) { "generation history is full" }
        file.parentFile?.let { parent ->
            check(parent.isDirectory || parent.mkdirs()) { "generation history directory is unavailable" }
        }
        val temporary = File(file.parentFile, "${file.name}.tmp")
        FileOutputStream(temporary).use { stream ->
            DataOutputStream(BufferedOutputStream(stream)).use { output ->
                output.writeUTF(MAGIC_V2)
                output.writeInt(records.size)
                records.forEach { writeRecord(output, it) }
                output.flush()
                stream.fd.sync()
            }
        }
        runCatching {
            Files.move(
                temporary.toPath(),
                file.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        }.getOrElse {
            Files.move(temporary.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun readRecord(input: DataInputStream, v2: Boolean): GenerationHistoryRecord {
        val id = input.readBoundedString("id")
        val createdAtMs = input.readLong()
        val promptBytes = input.readBoundedBytes("prompt")
        val mode = GenerationMode.entries.getOrNull(input.readInt()) ?: error("generation history mode is invalid")
        val temperature = input.readFloat()
        val topK = input.readInt()
        val samplingSeed = input.readLong()
        val maxNewBytes = input.readInt()
        val checkpointStep = input.readInt()
        val checkpointParameterHash = input.readBoundedString("checkpoint hash")
        val vocabulary = input.readInt()
        val tokens = input.readInt()
        val dimension = input.readInt()
        val feedForwardDimension = input.readInt()
        val layers = input.readInt()
        val heads = input.readInt()
        val checkpointFormat = if (v2) input.readBoundedString("checkpoint format") else "NPRTCKPTV2"
        val tokenizerKind = if (v2) input.readBoundedString("tokenizer kind") else "byte"
        val tokenizerHash = if (v2) input.readNullableString() else null
        val generatedTokenCount = if (v2) input.readInt() else -1
        val generatedBytes = input.readBoundedBytes("generated output")
        return GenerationHistoryRecord(
        id = id, createdAtMs = createdAtMs, promptBytes = promptBytes, mode = mode,
        temperature = temperature, topK = topK, samplingSeed = samplingSeed,
        maxNewBytes = maxNewBytes, checkpointStep = checkpointStep,
        checkpointParameterHash = checkpointParameterHash, vocabulary = vocabulary,
        tokens = tokens, dimension = dimension, feedForwardDimension = feedForwardDimension,
        layers = layers, heads = heads, generatedBytes = generatedBytes,
        elapsedMs = input.readLong(),
        backend = input.readBoundedString("backend"),
        qnnExecuteAttempts = input.readLong(),
        qnnExecuteSuccesses = input.readLong(),
        qnnExecuteFailures = input.readLong(),
        cpuFallback = input.readBoolean(),
        finite = input.readBoolean(),
        status = GenerationHistoryStatus.entries.getOrNull(input.readInt())
            ?: error("generation history status is invalid"),
        failureMessage = input.readNullableString(),
        checkpointFormat = checkpointFormat, tokenizerKind = tokenizerKind,
        tokenizerHash = tokenizerHash,
        generatedTokenCount = if (generatedTokenCount >= 0) generatedTokenCount else generatedBytes.size,
    ).also(::validate)
    }

    private fun writeRecord(output: DataOutputStream, record: GenerationHistoryRecord) {
        validate(record)
        output.writeBoundedString(record.id)
        output.writeLong(record.createdAtMs)
        output.writeBoundedBytes(record.promptBytes)
        output.writeInt(record.mode.ordinal)
        output.writeFloat(record.temperature)
        output.writeInt(record.topK)
        output.writeLong(record.samplingSeed)
        output.writeInt(record.maxNewBytes)
        output.writeInt(record.checkpointStep)
        output.writeBoundedString(record.checkpointParameterHash)
        output.writeInt(record.vocabulary)
        output.writeInt(record.tokens)
        output.writeInt(record.dimension)
        output.writeInt(record.feedForwardDimension)
        output.writeInt(record.layers)
        output.writeInt(record.heads)
        output.writeBoundedString(record.checkpointFormat)
        output.writeBoundedString(record.tokenizerKind)
        output.writeNullableString(record.tokenizerHash)
        output.writeInt(record.generatedTokenCount)
        output.writeBoundedBytes(record.generatedBytes)
        output.writeLong(record.elapsedMs)
        output.writeBoundedString(record.backend)
        output.writeLong(record.qnnExecuteAttempts)
        output.writeLong(record.qnnExecuteSuccesses)
        output.writeLong(record.qnnExecuteFailures)
        output.writeBoolean(record.cpuFallback)
        output.writeBoolean(record.finite)
        output.writeInt(record.status.ordinal)
        output.writeNullableString(record.failureMessage)
    }

    private fun validate(record: GenerationHistoryRecord) {
        require(record.id.isNotBlank() && record.id.length <= MAX_STRING_BYTES)
        require(record.createdAtMs >= 0)
        require(record.promptBytes.size <= MAX_BLOB_BYTES && record.generatedBytes.size <= MAX_BLOB_BYTES)
        require(record.temperature.isFinite())
        require(record.topK in 1..record.vocabulary && record.maxNewBytes in 1..1024)
        require(record.checkpointStep > 0)
        require(record.checkpointParameterHash.matches(Regex("fnv1a64:[0-9a-f]{16}")))
        require(record.vocabulary > 0 && record.tokens > 0 && record.dimension > 0 &&
            record.feedForwardDimension > 0 && record.layers > 0 && record.heads > 0)
        require(record.generatedTokenCount >= 0 && record.generatedTokenCount <= record.maxNewBytes)
        require(record.checkpointFormat == "NPRTCKPTV2" || record.checkpointFormat == "NPRTCKPTV3")
        require((record.vocabulary == 1024) == (record.tokenizerKind == "byte_bpe"))
        if (record.tokenizerKind == "byte_bpe")
            require(record.tokenizerHash?.matches(Regex("sha256:[0-9a-f]{64}")) == true)
        else require(record.tokenizerHash == null)
        require(record.elapsedMs >= 0 && record.qnnExecuteAttempts >= 0 &&
            record.qnnExecuteSuccesses >= 0 && record.qnnExecuteFailures >= 0)
        require(record.qnnExecuteSuccesses + record.qnnExecuteFailures <= record.qnnExecuteAttempts)
        if (record.status == GenerationHistoryStatus.SUCCESS) require(record.failureMessage == null)
    }

    private fun DataInputStream.readBoundedString(name: String): String {
        val size = readInt()
        require(size in 0..MAX_STRING_BYTES) { "generation history $name size is invalid" }
        val bytes = ByteArray(size).also(::readFully)
        return String(bytes, Charsets.UTF_8)
    }

    private fun DataInputStream.readBoundedBytes(name: String): ByteArray {
        val size = readInt()
        require(size in 0..MAX_BLOB_BYTES) { "generation history $name size is invalid" }
        return ByteArray(size).also(::readFully)
    }

    private fun DataInputStream.readNullableString(): String? = if (readBoolean()) readBoundedString("message") else null

    private fun DataOutputStream.writeBoundedString(value: String) {
        val bytes = value.toByteArray(Charsets.UTF_8)
        require(bytes.size <= MAX_STRING_BYTES)
        writeInt(bytes.size)
        write(bytes)
    }

    private fun DataOutputStream.writeBoundedBytes(value: ByteArray) {
        require(value.size <= MAX_BLOB_BYTES)
        writeInt(value.size)
        write(value)
    }

    private fun DataOutputStream.writeNullableString(value: String?) {
        writeBoolean(value != null)
        if (value != null) writeBoundedString(value)
    }

    private companion object {
        const val MAGIC_V1 = "PHONELM_GENERATION_HISTORY_V1"
        const val MAGIC_V2 = "PHONELM_GENERATION_HISTORY_V2"
        const val MAX_RECORDS = 10_000
        const val MAX_BLOB_BYTES = 16 * 1024 * 1024
        const val MAX_STRING_BYTES = 64 * 1024
    }
}
