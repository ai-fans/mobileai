package com.aosp.moblieai;

import androidx.activity.ComponentActivity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;

import com.aosp.moblieai.ondeviceai.ClassificationResult;
import com.aosp.moblieai.ondeviceai.ImageClassifier;

import java.io.InputStream;
import java.util.List;

public class MainActivity extends ComponentActivity {

    private ImageView imageView;
    private TextView resultText;
    private TextView perfText;
    private Button selectImageBtn;
    private Button cameraBtn;
    private Button classifyBtn;
    private Spinner modelSpinner;

    private Bitmap selectedBitmap = null;
    private ImageClassifier classifier = null;

    private static class ModelInfo {
        final String name;
        final String modelPath;
        final String labelPath;

        ModelInfo(String name, String modelPath, String labelPath) {
            this.name = name;
            this.modelPath = modelPath;
            this.labelPath = labelPath;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private final ModelInfo[] models = {
        new ModelInfo("MobileNetV2 (\u6d6e\u70b9)", "mobilenetv2.tflite", "labels_mobilenetv2.txt"),
        new ModelInfo("MobileNetV1 (\u91cf\u5316)", "mobilenet_v1_1.0_224_quant.tflite", "labels_mobilenet_quant_v1_224.txt")
    };

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
                                            resultText.setText("\u56fe\u7247\u5df2\u52a0\u8f7d\uff0c\u70b9\u51fb\u300c\u5f00\u59cb\u8bc6\u522b\u300d");
                                            perfText.setText("");
                                        } else {
                                            Toast.makeText(MainActivity.this,
                                                    "\u56fe\u7247\u52a0\u8f7d\u5931\u8d25\uff0c\u8bf7\u91cd\u8bd5", Toast.LENGTH_SHORT).show();
                                        }
                                    }
                                }
                            }
                        }
                    });

    // 系统相机拍照的 Launcher
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
        setupModelSpinner();
        initClassifier(0);
        setupListeners();
    }

    private void initViews() {
        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.resultText);
        perfText = findViewById(R.id.perfText);
        selectImageBtn = findViewById(R.id.selectImageBtn);
        cameraBtn = findViewById(R.id.cameraBtn);
        classifyBtn = findViewById(R.id.classifyBtn);
        modelSpinner = findViewById(R.id.modelSpinner);

        classifyBtn.setEnabled(false);
        resultText.setText("\u8bf7\u9009\u62e9\u4e00\u5f20\u56fe\u7247");
    }

    private void setupModelSpinner() {
        ArrayAdapter<ModelInfo> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, models);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(adapter);
        modelSpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, android.view.View view, int position, long id) {
                initClassifier(position);
                if (selectedBitmap != null) {
                    resultText.setText("\u6a21\u578b\u5df2\u5207\u6362\uff0c\u70b9\u51fb\u300c\u5f00\u59cb\u8bc6\u522b\u300d\u91cd\u65b0\u8bc6\u522b");
                    perfText.setText("");
                }
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        });
    }

    private void initClassifier(int modelIndex) {
        if (classifier != null) {
            classifier.close();
        }
        try {
            ModelInfo model = models[modelIndex];
            classifier = new ImageClassifier(this, model.modelPath, model.labelPath);
        } catch (Exception e) {
            Toast.makeText(this, "\u6a21\u578b\u52a0\u8f7d\u5931\u8d25\uff1a" + e.getMessage(), Toast.LENGTH_LONG).show();
            resultText.setText("\u274c \u6a21\u578b\u52a0\u8f7d\u5931\u8d25\uff1a" + e.getMessage());
        }
    }

    private void setupListeners() {
        selectImageBtn.setOnClickListener(v -> openGallery());
        cameraBtn.setOnClickListener(v -> cameraLauncher.launch(null));
        classifyBtn.setOnClickListener(v -> {
            if (selectedBitmap != null) {
                classifyImage(selectedBitmap);
            } else {
                Toast.makeText(MainActivity.this, "\u8bf7\u5148\u9009\u62e9\u56fe\u7247", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        pickImageLauncher.launch(intent);
    }

    private Bitmap getBitmapFromUri(Uri uri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            return BitmapFactory.decodeStream(inputStream);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void classifyImage(Bitmap bitmap) {
        if (classifier == null) {
            resultText.setText("\u274c \u6a21\u578b\u672a\u52a0\u8f7d");
            return;
        }

        resultText.setText("\u23f3 \u6b63\u5728\u8bc6\u522b...");
        perfText.setText("");
        classifyBtn.setEnabled(false);

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    var result = classifier.classify(bitmap, 5);
                    List<ClassificationResult> topResults = result.getFirst();
                    long inferenceTime = result.getSecond();
                    long modelSize = classifier.getModelSizeKB();
                    String modelType = classifier.isQuantizedModel() ? "INT8\u91cf\u5316" : "FP32\u6d6e\u70b9";

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            resultText.setText(buildResultString(topResults));
                            perfText.setText(String.format(
                                    "\u23f1 \u63a8\u7406\u8017\u65f6: %dms | \ud83d\udce6 \u6a21\u578b: %dKB | \ud83d\udd27 %s",
                                    inferenceTime, modelSize, modelType));
                            classifyBtn.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            resultText.setText("\u274c \u8bc6\u522b\u5931\u8d25\uff1a" + e.getMessage());
                            classifyBtn.setEnabled(true);
                        }
                    });
                    e.printStackTrace();
                }
            }
        }).start();
    }

    private String buildResultString(List<ClassificationResult> results) {
        if (results == null || results.isEmpty()) return "\u672a\u8bc6\u522b\u5230\u7ed3\u679c";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < results.size(); i++) {
            ClassificationResult r = results.get(i);
            float pct = r.getConfidence() * 100;
            String prefix = (i == 0) ? "\ud83e\udd47" : (i == 1) ? "\ud83e\udd48" : (i == 2) ? "\ud83e\udd49" : "  " + (i + 1) + ".";
            sb.append(String.format("%s %-20s %.2f%%\n", prefix, r.getLabel(), pct));
        }
        return sb.toString().trim();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (classifier != null) {
            classifier.close();
        }
    }
}
