package com.aosp.moblieai;

import androidx.activity.ComponentActivity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;

import com.aosp.moblieai.ondeviceai.ImageClassifier;

import java.io.InputStream;

public class MainActivity extends ComponentActivity {

    private ImageView imageView;
    private TextView resultText;
    private Button selectImageBtn;
    private Button cameraBtn;
    private Button classifyBtn;

    private Bitmap selectedBitmap = null;
    private ImageClassifier classifier = null;

    // 从相册选择图片的 Launcher
    private final ActivityResultLauncher<Intent> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                    new ActivityResultCallback<ActivityResult>() {
                        @Override
                        public void onActivityResult(ActivityResult result) {
                            if (result.getResultCode() == RESULT_OK) {
                                Intent data = result.getData();
                                if (data != null) {
                                    Uri imageUri = data.getData();
                                    if (imageUri != null) {
                                        selectedBitmap = getBitmapFromUri(imageUri);
                                        if (selectedBitmap != null) {
                                            imageView.setImageBitmap(selectedBitmap);
                                            classifyBtn.setEnabled(true);
                                            resultText.setText("图片已加载，点击「开始识别」");
                                        } else {
                                            Toast.makeText(MainActivity.this,
                                                    "图片加载失败，请重试", Toast.LENGTH_SHORT).show();
                                        }
                                    }
                                }
                            }
                        }
                    });

    // 系统相机拍照的 Launcher（返回缩略图 Bitmap）
    private final ActivityResultLauncher<Void> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicturePreview(),
                    new ActivityResultCallback<Bitmap>() {
                        @Override
                        public void onActivityResult(Bitmap bitmap) {
                            if (bitmap != null) {
                                selectedBitmap = bitmap;
                                imageView.setImageBitmap(bitmap);
                                classifyImage(bitmap);
                            }
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initViews();
        initClassifier();
        setupListeners();
    }

    private void initViews() {
        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.resultText);
        selectImageBtn = findViewById(R.id.selectImageBtn);
        cameraBtn = findViewById(R.id.cameraBtn);
        classifyBtn = findViewById(R.id.classifyBtn);

        classifyBtn.setEnabled(false);
        resultText.setText("请选择一张图片");
    }

    private void initClassifier() {
        try {
            classifier = new ImageClassifier(this);
        } catch (Exception e) {
            Toast.makeText(this, "模型加载失败：" + e.getMessage(), Toast.LENGTH_LONG).show();
            resultText.setText("❌ 模型加载失败");
        }
    }

    private void setupListeners() {
        // 选择图片按钮
        selectImageBtn.setOnClickListener(v -> openGallery());

        // 拍照识别按钮
        cameraBtn.setOnClickListener(v -> cameraLauncher.launch(null));

        // 识别按钮
        classifyBtn.setOnClickListener(v -> {
            if (selectedBitmap != null) {
                classifyImage(selectedBitmap);
            } else {
                Toast.makeText(MainActivity.this, "请先选择图片", Toast.LENGTH_SHORT).show();
            }
        });
    }

    // 打开相册
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        pickImageLauncher.launch(intent);
    }

    // 从 Uri 读取 Bitmap
    private Bitmap getBitmapFromUri(Uri uri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            return BitmapFactory.decodeStream(inputStream);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // 执行图像分类
    private void classifyImage(Bitmap bitmap) {
        if (classifier == null) {
            resultText.setText("❌ 模型未加载");
            return;
        }

        // 显示加载状态
        resultText.setText("⏳ 正在识别...");
        classifyBtn.setEnabled(false);

        // 在子线程执行推理，避免阻塞 UI
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    String result = classifier.classify(bitmap);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            resultText.setText(result);
                            classifyBtn.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            resultText.setText("❌ 识别失败：" + e.getMessage());
                            classifyBtn.setEnabled(true);
                        }
                    });
                    e.printStackTrace();
                }
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 释放模型资源
        if (classifier != null) {
            classifier.close();
        }
    }
}