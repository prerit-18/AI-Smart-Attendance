import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

# Add base directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IMAGE_SIZE, FACE_MODEL_DIR

# Global variable to cache the model in memory for performance (app.py reuse)
_MODEL_CACHE = {}

def build_custom_cnn(input_shape=(160, 160, 3), embedding_dim=128, use_regularization=True):
    """
    Builds a custom CNN model from scratch (Course 4).
    demonstrates: Convolution, Pooling, Dense Layers, Batch Normalization, and Dropout (Course 2).
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Conv Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    if use_regularization:
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Conv Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    if use_regularization:
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Conv Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    if use_regularization:
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Dense Projection
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    if use_regularization:
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    if use_regularization:
        model.add(layers.Dropout(0.5))  # Dropout for optimization (Course 2)
        
    model.add(layers.Dense(embedding_dim))
    
    # L2 normalize embeddings so cosine similarity can be calculated via simple dot product
    model.add(layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    
    return model

def build_mobilenet_extractor(input_shape=(160, 160, 3), embedding_dim=128):
    """
    Builds a Transfer Learning model using MobileNetV2 pre-trained on ImageNet (Course 4).
    The base layers are frozen, acting as a feature extractor.
    We extract the pooling layer features directly and L2 normalize them to enable
    high-accuracy zero-shot face comparison out-of-the-box.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze weights for transfer learning
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    # Ensure inference mode is active for BatchNormalization layers in MobileNetV2
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # L2 normalization layer
    normalized_outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    
    model = models.Model(inputs, normalized_outputs)
    return model

def get_face_model(model_type="mobilenet", force_reload=False):
    """
    Loads and returns the face recognition feature extractor model.
    Caches the model in memory.
    """
    global _MODEL_CACHE
    cache_key = model_type
    
    if cache_key in _MODEL_CACHE and not force_reload:
        return _MODEL_CACHE[cache_key]
        
    model_path = os.path.join(FACE_MODEL_DIR, f"{model_type}_face_model.h5")
    
    # Try to load custom trained weights if they exist, otherwise initialize from backbone
    if os.path.exists(model_path):
        try:
            # Custom lambda layers may require custom_objects, or we load as saved model
            model = models.load_model(model_path, compile=False)
            _MODEL_CACHE[cache_key] = model
            return model
        except Exception as e:
            print(f"Error loading saved model {model_path}: {e}. Initializing fresh weights.")
            
    # Fresh Initialization
    if model_type == "mobilenet":
        model = build_mobilenet_extractor(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    elif model_type == "custom_cnn":
        model = build_custom_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), use_regularization=True)
    elif model_type == "custom_cnn_baseline":
        model = build_custom_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), use_regularization=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    _MODEL_CACHE[cache_key] = model
    return model

def get_embedding(model, preprocessed_image):
    """
    Runs forward propagation to generate a 128-dimensional face embedding.
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed numpy face image of shape IMAGE_SIZE
    Returns:
        1D numpy array of size 128 representing face embedding
    """
    if preprocessed_image is None or model is None:
        return None
        
    # Add batch dimension: (1, H, W, C)
    if len(preprocessed_image.shape) == 3:
        batch_img = np.expand_dims(preprocessed_image, axis=0)
    else:
        batch_img = preprocessed_image
        
    # Forward pass
    embedding = model(batch_img, training=False)
    return embedding.numpy()[0]

def compute_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two L2 normalized embeddings.
    Since embeddings are already L2 normalized, cosine similarity is just the dot product!
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    if len(embedding1) != len(embedding2):
        return 0.0
    return float(np.dot(embedding1, embedding2))
