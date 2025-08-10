import tensorflow as tf
import os

def convert_model():
    """Convert old Keras model to new format"""
    
    MODEL_PATH = "model/ae_user_s036.h5"
    NEW_MODEL_PATH = "model/ae_user_s036_converted.h5"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    
    try:
        # Load model without compile
        print("Loading model without compile...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Recompile with modern syntax
        print("Recompiling model...")
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mse']
        )
        
        # Save converted model
        print(f"Saving converted model to {NEW_MODEL_PATH}...")
        model.save(NEW_MODEL_PATH)
        
        print("✅ Model converted successfully!")
        print(f"Update your MODEL_PATH to: {NEW_MODEL_PATH}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == "__main__":
    convert_model()