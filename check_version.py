# Quick verification script - Run this to check which version you have
# Save as check_version.py and run: python check_version.py

import os
import sys

def check_file_version():
    """Check which version of the trading bot you're running"""
    
    print("🔍 Checking Trading Bot Version...")
    print("=" * 50)
    
    # Check if the file exists
    file_path = "optimized_model_trainer.py"
    if not os.path.exists(file_path):
        print("❌ optimized_model_trainer.py not found!")
        return
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for enhanced version indicators
    enhanced_indicators = {
        "Enhanced Trading Bot": "Enhanced version header",
        "import ta": "Technical Analysis library",
        "from imblearn.over_sampling import SMOTE": "SMOTE class balancing",
        "EnhancedFeatureEngine": "Enhanced feature engineering class",
        "train_ensemble_enhanced": "Enhanced ensemble method",
        "RobustScaler": "Enhanced scaler",
        "ExtraTreesClassifier": "Extra Trees classifier",
        "meta_ensemble": "Meta-ensemble functionality",
        "TARGET ACHIEVED": "Target achievement detection"
    }
    
    print("📋 Feature Detection:")
    found_count = 0
    total_count = len(enhanced_indicators)
    
    for indicator, description in enhanced_indicators.items():
        if indicator in content:
            print(f"   ✅ {description}")
            found_count += 1
        else:
            print(f"   ❌ {description}")
    
    print(f"\n📊 Enhanced Features Found: {found_count}/{total_count}")
    
    if found_count >= 7:
        print("🎉 ENHANCED VERSION DETECTED!")
        print("   Your file has the enhanced features.")
        print("   Issue might be elsewhere...")
    elif found_count >= 3:
        print("⚠️  PARTIAL ENHANCEMENT DETECTED")
        print("   Some features are enhanced, but not all.")
        print("   Recommend complete file replacement.")
    else:
        print("❌ OLD VERSION DETECTED")
        print("   You're still running the original version.")
        print("   Need to replace the entire file.")
    
    # Check file size (enhanced version should be much larger)
    file_size = os.path.getsize(file_path)
    print(f"\n📏 File Size: {file_size:,} bytes")
    
    if file_size > 30000:  # Enhanced version should be 30KB+
        print("   ✅ File size suggests enhanced version")
    else:
        print("   ❌ File size suggests original version")
    
    print(f"\n🔧 Recommendations:")
    if found_count < 7:
        print("   1. Backup current file: copy optimized_model_trainer.py backup.py")
        print("   2. Replace entire file with enhanced version")
        print("   3. Ensure all imports are working: pip install ta imblearn")
        print("   4. Re-run training")
    else:
        print("   1. Check if all imports installed correctly")
        print("   2. Look for runtime errors in the log")
        print("   3. Verify data quality and market conditions")

if __name__ == "__main__":
    check_file_version()