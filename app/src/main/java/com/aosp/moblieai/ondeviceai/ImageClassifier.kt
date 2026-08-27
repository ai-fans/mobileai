package com.aosp.moblieai.ondeviceai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class ClassificationResult(
    val label: String,
    val confidence: Float
)

class ImageClassifier @JvmOverloads constructor(
    private val context: Context,
    private val modelPath: String = "mobilenetv2.tflite",
    private val labelPath: String = "labels_mobilenetv2.txt",
    private val numThreads: Int = 4
) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var inputSize: Int = 0
    private var numClasses: Int = 0
    private var isQuantized: Boolean = false

    init {
        labels = loadLabels()
        loadModel()
    }

    private fun loadLabels(): List<String> {
        return context.assets.open(labelPath).bufferedReader().useLines { lines ->
            lines.toList()
        }
    }

    private fun loadModel() {
        val modelBuffer = loadModelFile()
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
        }
        interpreter = Interpreter(modelBuffer, options)

        // 从输入 Tensor 获取输入尺寸和数据类型
        val inputTensor = interpreter!!.getInputTensor(0)
        val inputShape = inputTensor.shape()
        inputSize = inputShape[1]
        isQuantized = inputTensor.dataType() == DataType.UINT8

        val outputShape = interpreter!!.getOutputTensor(0).shape()
        numClasses = outputShape[1]
    }

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // 执行推理：输入 Bitmap，输出 Top-K 分类结果和推理耗时
    fun classify(bitmap: Bitmap, topK: Int = 5): Pair<List<ClassificationResult>, Long> {
        // 1. 缩放到模型所需输入尺寸
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // 2. 根据模型类型分配输入 Buffer（量化: 1 byte/像素, 浮点: 4 bytes/像素）
        val inputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(inputSize * inputSize * 3)
        } else {
            ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        }
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resizedBitmap.getPixel(x, y)
                if (isQuantized) {
                    // 量化模型：直接写入像素原始字节值（0-255）
                    inputBuffer.put(Color.red(pixel).toByte())
                    inputBuffer.put(Color.green(pixel).toByte())
                    inputBuffer.put(Color.blue(pixel).toByte())
                } else {
                    // 浮点模型：归一化到 [0, 1]
                    inputBuffer.putFloat(Color.red(pixel) / 255.0f)
                    inputBuffer.putFloat(Color.green(pixel) / 255.0f)
                    inputBuffer.putFloat(Color.blue(pixel) / 255.0f)
                }
            }
        }
        inputBuffer.rewind()

        // 3. 准备输出 Buffer
        val outputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(numClasses)
        } else {
            ByteBuffer.allocateDirect(4 * numClasses)
        }
        outputBuffer.order(ByteOrder.nativeOrder())

        // 4. 执行推理并计时
        val startTime = System.currentTimeMillis()
        interpreter?.run(inputBuffer, outputBuffer)
        val inferenceTime = System.currentTimeMillis() - startTime

        // 5. 读取输出并转为 float 数组
        outputBuffer.rewind()
        val scores = FloatArray(numClasses)

        val outputTensor = interpreter!!.getOutputTensor(0)
        if (outputTensor.dataType() == DataType.FLOAT32) {
            outputBuffer.asFloatBuffer().get(scores)
        } else {
            // 量化输出需要反量化：float_value = (uint8_value - zero_point) * scale
            val quantParams = outputTensor.quantizationParams()
            val scale = quantParams.scale
            val zeroPoint = quantParams.zeroPoint
            val rawBytes = ByteArray(numClasses)
            outputBuffer.get(rawBytes)
            for (i in 0 until numClasses) {
                scores[i] = ((rawBytes[i].toInt() and 0xFF) - zeroPoint) * scale
            }
        }

        // 6. 选出 Top-K 结果
        val topResults = mutableListOf<ClassificationResult>()
        val usedIndices = mutableSetOf<Int>()
        for (k in 0 until minOf(topK, numClasses)) {
            var maxIndex = -1
            var maxScore = -Float.MAX_VALUE
            for (i in 0 until numClasses) {
                if (i !in usedIndices && scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxIndex = i
                }
            }
            if (maxIndex >= 0) {
                usedIndices.add(maxIndex)
                val label = if (maxIndex < labels.size) labels[maxIndex] else "unknown"
                topResults.add(ClassificationResult(label, maxScore))
            }
        }

        return Pair(topResults, inferenceTime)
    }

    // 获取模型文件大小（KB）
    fun getModelSizeKB(): Long {
        return try {
            val afd = context.assets.openFd(modelPath)
            afd.declaredLength / 1024
        } catch (e: Exception) {
            -1
        }
    }

    fun getModelName(): String = modelPath

    fun isQuantizedModel(): Boolean = isQuantized

    fun close() {
        interpreter?.close()
    }
}
