# Simple Keras Check - Fixed Version
# Save as 'check_keras.py' and run it

def check_tensorflow_keras():
    """Check TensorFlow and Keras availability with better error handling"""
    
    print("🔍 Checking TensorFlow and Keras...")
    print("-" * 40)
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        
        # Try different ways to get version
        version = "Unknown"
        try:
            version = tf.__version__
        except AttributeError:
            try:
                version = tf.version.VERSION
            except AttributeError:
                try:
                    version = tf.VERSION
                except AttributeError:
                    pass
        
        print(f"✅ TensorFlow: Available (Version: {version})")
        tf_available = True
        
    except ImportError as e:
        print(f"❌ TensorFlow: Not available ({e})")
        tf_available = False
    
    # Check Keras (multiple ways)
    keras_available = False
    keras_source = ""
    
    if tf_available:
        # Try Keras via TensorFlow
        try:
            from tensorflow import keras
            print("✅ Keras: Available via TensorFlow")
            keras_available = True
            keras_source = "tensorflow.keras"
        except ImportError:
            pass
    
    if not keras_available:
        # Try standalone Keras
        try:
            import keras
            print("✅ Keras: Available as standalone")
            keras_available = True
            keras_source = "standalone keras"
        except ImportError:
            print("❌ Keras: Not available")
    
    # Check scikit-learn
    try:
        import sklearn
        sklearn_version = sklearn.__version__
        print(f"✅ Scikit-learn: Available (Version: {sklearn_version})")
        sklearn_available = True
    except ImportError:
        print("❌ Scikit-learn: Not available")
        sklearn_available = False
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"TensorFlow: {'✅ Available' if tf_available else '❌ Missing'}")
    print(f"Keras: {'✅ Available (' + keras_source + ')' if keras_available else '❌ Missing'}")
    print(f"Scikit-learn: {'✅ Available' if sklearn_available else '❌ Missing'}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if tf_available and keras_available and sklearn_available:
        print("🎉 All ML libraries are working! Your bot has full ML capabilities.")
    
    elif tf_available and sklearn_available:
        print("✅ Core ML libraries working. Your bot will work great!")
        if not keras_available:
            print("📝 Optional: Install Keras for advanced neural networks:")
            print("   pip install keras")
    
    elif sklearn_available:
        print("✅ Basic ML available with scikit-learn. Core bot functions will work.")
        print("📝 For deep learning, install TensorFlow:")
        print("   pip install tensorflow")
    
    else:
        print("⚠️ Install basic ML libraries:")
        print("   pip install scikit-learn tensorflow")
    
    return {
        'tensorflow': tf_available,
        'keras': keras_available, 
        'sklearn': sklearn_available
    }

if __name__ == "__main__":
    check_tensorflow_keras()