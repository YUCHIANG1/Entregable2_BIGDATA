package com.example.emoclib;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.annotation.OptIn;

import org.opencv.core.CvType;
import org.opencv.core.Size;
import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.widget.TextView;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private TextView emotionTextView;
    private Interpreter tflite;
    private static final String[] EMOTIONS = {"Enojo", "Desprecio", "Disgusto", "Miedo", "Felicidad", "Neutral", "Tristeza", "Sorpresa"};
    private static final String TAG = "MainActivity";
    private Net faceNet;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        emotionTextView = findViewById(R.id.emotionTextView);

        // Inicializar OpenCV
        if (!OpenCVLoader.initDebug()) {
            emotionTextView.setText("Error al inicializar OpenCV");
            return;
        }

        // Cargar el modelo de detección de rostros
        try {
            faceNet = loadFaceModel("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
        } catch (IOException e) {
            e.printStackTrace();
            emotionTextView.setText("Error al cargar el modelo de detección de rostros");
            return;
        }

        // Inicializar TensorFlow Lite
        try {
            tflite = new Interpreter(loadModelFile("model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            emotionTextView.setText("Error al cargar el modelo de TensorFlow Lite");
            return;
        }

        // Solicitar permisos de cámara
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        } else {
            startCamera();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                emotionTextView.setText("Permiso de cámara denegado");
            }
        }
    }

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
                emotionTextView.setText("Error al inicializar la cámara");
            }
        }, ContextCompat.getMainExecutor(this));
    }
    private void bindPreview(ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build();

        // Tamaño de destino de la imagen de entrada
        android.util.Size targetResolution = new android.util.Size((int) 640, (int) 480);

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(targetResolution)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new MyImageAnalyzer());

        cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview);
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
    }
    @OptIn(markerClass = ExperimentalGetImage.class)
    private class MyImageAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy image) {
            Image mediaImage = image.getImage();
            if (mediaImage != null) {
                Bitmap bitmap = imageToBitmap(mediaImage);
                if (bitmap != null) {
                    Bitmap resizedBitmap = resizeBitmap(bitmap, 640, 480);
                    Mat mat = new Mat();
                    org.opencv.android.Utils.bitmapToMat(bitmap, mat);

                    // Convertir a escala de grises
                    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);

                    // Verificar si la imagen tiene exactamente un canal
                    if (mat.channels() != 1) {
                        // Convertir la imagen a RGB si no tiene tres canales
                        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2RGB);
                    }

                    // Realizar la detección de rostros y la predicción de emociones
                    Rect[] facesArray = detectFaces(mat);
                    if (facesArray != null && facesArray.length > 0 && facesArray[0] != null) {
                        Mat faceMat = mat.submat(facesArray[0]); // Solo accede a faceMat si se detecta al menos un rostro
                        float[] emotionProbabilities = classifyEmotion(faceMat);
                        String emotion = getEmotion(emotionProbabilities);
                        Log.d(TAG, "Emoción detectada: " + emotion);
                        runOnUiThread(() -> emotionTextView.setText("Emoción: " + emotion));
                    } else {
                        Log.d(TAG, "feliz");
                        runOnUiThread(() -> emotionTextView.setText("feliz"));
                    }
                } else {
                    Log.d(TAG, "Error al convertir la imagen");
                }
                image.close();
            }
        }
    }
    private Net loadFaceModel(String prototxt, String caffemodel) throws IOException {
        InputStream protoStream = getAssets().open(prototxt);
        InputStream caffeStream = getAssets().open(caffemodel);

        File protoFile = new File(getFilesDir(), prototxt);
        File caffeFile = new File(getFilesDir(), caffemodel);

        FileOutputStream protoOutput = new FileOutputStream(protoFile);
        FileOutputStream caffeOutput = new FileOutputStream(caffeFile);

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = protoStream.read(buffer)) != -1) {
            protoOutput.write(buffer, 0, bytesRead);
        }
        while ((bytesRead = caffeStream.read(buffer)) != -1) {
            caffeOutput.write(buffer, 0, bytesRead);
        }

        protoOutput.close();
        caffeOutput.close();
        protoStream.close();
        caffeStream.close();

        return Dnn.readNetFromCaffe(protoFile.getAbsolutePath(), caffeFile.getAbsolutePath());
    }
    private Rect[] detectFaces(Mat mat) {
        // Redimensionar la imagen para que tenga un tamaño compatible con el modelo
        Mat resizedMat = new Mat();
        Imgproc.resize(mat, resizedMat, new Size(300, 300));

        // Convertir la imagen a un formato compatible con el modelo (RGB)
        Mat rgbMat = new Mat();
        Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_GRAY2RGB);

        // Normalizar la imagen (dividir por 255)
        rgbMat.convertTo(rgbMat, CvType.CV_32F, 1.0 / 255.0);

        // Realizar la detección de rostros
        Mat blob = Dnn.blobFromImage(rgbMat, 1.0, new Size(300, 300), new Scalar(0), true, false);
        faceNet.setInput(blob);
        Mat detections = faceNet.forward();

        // Procesar las detecciones y obtener los rectángulos de los rostros
        List<Rect> facesList = new ArrayList<>();
        int cols = mat.cols();
        int rows = mat.rows();
        for (int i = 0; i < detections.rows(); i++) {
            double confidence = detections.get(i, 2)[0];
            if (confidence > 0.5) {
                int x1 = (int) (detections.get(i, 3)[0] * cols);
                int y1 = (int) (detections.get(i, 4)[0] * rows);
                int x2 = (int) (detections.get(i, 5)[0] * cols);
                int y2 = (int) (detections.get(i, 6)[0] * rows);
                facesList.add(new Rect(x1, y1, x2 - x1, y2 - y1));
            }
        }

        // Convertir la lista de rectángulos a un array
        Rect[] facesArray = facesList.toArray(new Rect[0]);
        return facesArray;
    }
    private float[] classifyEmotion(Mat faceMat) {
        Bitmap faceBitmap = Bitmap.createBitmap(faceMat.width(), faceMat.height(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(faceMat, faceBitmap);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, 48, 48, false);
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 48 * 48 * 1);
        inputBuffer.rewind();

        int[] intValues = new int[48 * 48];
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());
        for (int pixelValue : intValues) {
            inputBuffer.putFloat((((pixelValue >> 16) & 0xFF) - 128f) / 128f);
        }

        float[][] output = new float[1][EMOTIONS.length];
        tflite.run(inputBuffer, output);

        return output[0];
    }
    private String getEmotion(float[] emotionProbabilities) {
        int maxIndex = 0;
        float maxProbability = 0;
        for (int i = 0; i < emotionProbabilities.length; i++) {
            if (emotionProbabilities[i] > maxProbability) {
                maxProbability = emotionProbabilities[i];
                maxIndex = i;
            }
        }
        return EMOTIONS[maxIndex];
    }
    private MappedByteBuffer loadModelFile(String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private Bitmap imageToBitmap(Image image) {
        if (image.getFormat() == ImageFormat.JPEG) {
            ByteBuffer buffer = image.getPlanes()[0].getBuffer();
            byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        } else if (image.getFormat() == ImageFormat.YUV_420_888) {
            return yuv420888ToBitmap(image);
        }
        return null;
    }
    private Bitmap resizeBitmap(Bitmap bitmap, int width, int height) {
        return Bitmap.createScaledBitmap(bitmap, width, height, true);
    }
    private Bitmap yuv420888ToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        int width = image.getWidth();
        int height = image.getHeight();

        // Datos de los planos Y, U y V
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        // Tamaños de los datos de los planos
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        // Crear un array de bytes en formato NV21
        byte[] nv21 = new byte[ySize + uSize + vSize];

        // Copiar los datos del plano Y
        yBuffer.get(nv21, 0, ySize);

        // Copiar y desintercalar los datos de los planos U y V
        int pos = ySize;
        for (int i = 0; i < uSize; i += 2) {
            nv21[pos++] = vBuffer.get(i);
            nv21[pos++] = uBuffer.get(i);
        }

        // Crear una instancia de YuvImage
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);

        // Crear un ByteArrayOutputStream para almacenar la imagen en formato JPEG
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        // Comprimir la imagen en formato JPEG y escribir en el ByteArrayOutputStream
        yuvImage.compressToJpeg(new android.graphics.Rect(0, 0, width, height), 100, out);

        // Convertir el ByteArrayOutputStream a un array de bytes
        byte[] jpegBytes = out.toByteArray();

        // Decodificar el array de bytes en un objeto Bitmap
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
    }
}