from tflite_model_maker import image_classifier, config
from tflite_model_maker.image_classifier import DataLoader

# Load input data specific to an on-device ML app.
data = DataLoader.from_folder('./test-models/')
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data, batch_size = 1)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model.
config = config.QuantizationConfig.for_dynamic()
model.export(export_dir='./tflite_models/', tflite_filename='model.tflite', quantization_config=config)