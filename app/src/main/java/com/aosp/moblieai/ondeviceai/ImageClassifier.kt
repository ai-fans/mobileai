package com.aosp.moblieai.ondeviceai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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

    init {
        labels = loadLabels()
        loadModel()
    }

    // 从 assets 加载标签文件
    private fun loadLabels(): List<String> {
        return context.assets.open(labelPath).bufferedReader().useLines { lines ->
            lines.toList()
        }
    }

    // 从 assets 加载模型文件并初始化 Interpreter
    private fun loadModel() {
        val modelBuffer = loadModelFile()
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
        }
        interpreter = Interpreter(modelBuffer, options)

        // 从模型的输入/输出 Tensor 动态获取尺寸
        val inputShape = interpreter!!.getInputTensor(0).shape()
        // 输入形状通常为 [1, height, width, channels]
        inputSize = inputShape[1]

        val outputShape = interpreter!!.getOutputTensor(0).shape()
        // 输出形状通常为 [1, numClasses]
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

    // 执行推理：输入 Bitmap，输出分类结果
    fun classify(bitmap: Bitmap): String {
        // 1. 将 Bitmap 缩放到模型所需输入尺寸，并写入 ByteBuffer
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3) // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder())
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resizedBitmap.getPixel(x, y)
                // 归一化到 0-1 范围
                inputBuffer.putFloat(Color.red(pixel) / 255.0f)
                inputBuffer.putFloat(Color.green(pixel) / 255.0f)
                inputBuffer.putFloat(Color.blue(pixel) / 255.0f)
            }
        }
        inputBuffer.rewind()

        // 2. 准备输出 Tensor（ByteBuffer 格式）
        val outputBuffer = ByteBuffer.allocateDirect(4 * numClasses)
        outputBuffer.order(ByteOrder.nativeOrder())

        // 3. 执行推理
        interpreter?.run(inputBuffer, outputBuffer)

        // 4. 读取输出并找到置信度最高的分类
        outputBuffer.rewind()
        val results = FloatArray(numClasses)
        outputBuffer.asFloatBuffer().get(results)
        var maxIndex = 0
        var maxConfidence = 0.0f
        for (i in results.indices) {
            if (results[i] > maxConfidence) {
                maxConfidence = results[i]
                maxIndex = i
            }
        }

        // 从标签列表中查找分类名称
        val confidence = String.format("%.2f", maxConfidence * 100)
        val label = if (maxIndex < labels.size) labels[maxIndex] else "unknown"
        return "$label (置信度: ${confidence}%)"
    }

    fun close() {
        interpreter?.close()
    }
}