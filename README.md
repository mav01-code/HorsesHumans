# Horse or Human Classifier using CNN

This project is a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images as either **Horse** or **Human**. It includes data preprocessing, model training, saving, and prediction for new images.

### 1. Install Requirements

```bash
pip install tensorflow numpy
```

### 2. Train the Model

The model uses `ImageDataGenerator` for image loading and preprocessing.

```python
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
```

### 3. Save the Trained Model

```python
model.save("horse_human_classifier.h5")
```

---

## Make Predictions on New Images

Use the `predict_image()` function to test a new image:

```python
predict_image("horse.jpg")
```

**Sample Output:**
```
horse.jpg --> Horse 🐎
```

---

## Model Architecture

- 3 Convolutional Layers with MaxPooling
- Flatten Layer
- Dense Layer with 512 Units (ReLU)
- Output Layer with 1 Unit (Sigmoid)

---

## Dependencies

- TensorFlow  
- NumPy

---

## Notes

- Input images must be RGB and resized to 150x150 pixels.
- Images are normalized (rescaled to range [0, 1]).
- If prediction < 0.5 → Horse, else → Human.
