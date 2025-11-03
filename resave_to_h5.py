import tensorflow as tf
import os

print("="*60)
print("Model Compatibility Conversion Script (.keras -> .h5)")
print(f"Using TensorFlow version: {tf.__version__}")
print("="*60)
print("\nIMPORTANT: Run this script in the EXACT SAME Python environment")
print("that was used to originally train and save 'final_model.keras'.\n")


KERAS_MODEL_PATH = 'final1/weight/final_model.keras'
H5_MODEL_PATH = 'final1/weight/final_model_compatible.h5'

if not os.path.exists(KERAS_MODEL_PATH):
    print(f"❌ ERROR: Original model not found at '{KERAS_MODEL_PATH}'")
    print("   Please make sure you are running this script from the project root.")
    exit(1)

print(f"➡️  Step 1: Loading original model from '{KERAS_MODEL_PATH}'...")
try:
    # We must load with safe_mode=False if the original model has lambda layers
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False, safe_mode=False)
    print("   ✅ Original model loaded successfully.")
    model.summary()
except Exception as e:
    print(f"\n   ❌ ERROR loading model: {e}")
    print("\n   Could not load the model. This is expected if you are not")
    print("   in the original training environment. Please switch to the correct")
    print("   Python/TensorFlow environment and try again.")
    exit(1)


print(f"\n➡️  Step 2: Saving compatible model to '{H5_MODEL_PATH}'...")
try:
    # Save in the legacy HDF5 format. It is more portable across versions.
    model.save(H5_MODEL_PATH, save_format='h5')
    print(f"   ✅ Model re-saved successfully to '{H5_MODEL_PATH}'")
except Exception as e:
    print(f"\n   ❌ ERROR saving model: {e}")
    exit(1)

print("\n" + "="*60)
print("🎉 SUCCESS!")
print(f"A new compatible model file has been created: '{H5_MODEL_PATH}'")
print("\nYou can now commit and push this new .h5 file.")
print("The deployment scripts will automatically use it.")
print("="*60)
