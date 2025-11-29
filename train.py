import os
import shutil
import zipfile
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- Configuration ---
MODEL_FILE = 'best_model.h5'
IMG_SIZE = 224
BATCH_SIZE = 32
RAW_DATA_DIR = 'dataset/dataset2-master/dataset2-master/images'
CLEAN_DATA_DIR = 'processed_data' # New clean folder

def setup_kaggle_api():
    if not os.path.exists('kaggle.json'):
        print("❌ kaggle.json not found. Please place it in the project directory.")
        exit()
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    shutil.move('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    print("✅ Kaggle API configured.")

def download_dataset():
    if os.path.exists(RAW_DATA_DIR):
        print("✅ Raw dataset already exists. Skipping download.")
        return
    
    print("\n--- Step 2: Downloading Dataset ---")
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('paultimothymooney/blood-cells', path='.', unzip=False)
    
    print("Extracting dataset...")
    with zipfile.ZipFile('blood-cells.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')
    os.remove('blood-cells.zip')
    print("✅ Dataset extracted.")

def organize_and_balance_data():
    """
    Reorganizes data into a clean structure:
    processed_data/
       train/
          LYMPHOCYTE/
          NEUTROPHIL/
       test/
          LYMPHOCYTE/
          NEUTROPHIL/
    """
    print("\n--- Step 2.5: Organizing & Checking Balance ---")
    
    classes = ['LYMPHOCYTE', 'NEUTROPHIL']
    subsets = ['TRAIN', 'TEST']
    
    # Create clean directories
    for subset in subsets:
        for cls in classes:
            os.makedirs(os.path.join(CLEAN_DATA_DIR, subset, cls), exist_ok=True)

    # Move/Copy files and Count
    for subset in subsets:
        print(f"\nProcessing {subset} set:")
        for cls in classes:
            # Path to the raw kaggle extraction
            src_dir = os.path.join(RAW_DATA_DIR, subset, cls)
            dest_dir = os.path.join(CLEAN_DATA_DIR, subset, cls)
            
            # Get all images
            images = glob.glob(os.path.join(src_dir, "*.jpeg")) + glob.glob(os.path.join(src_dir, "*.jpg"))
            
            # Copy files to new destination
            count = 0
            for img in images:
                shutil.copy(img, dest_dir)
                count += 1
                
            print(f"   -> Class {cls}: {count} images found.")
            
    print(f"\n✅ Data successfully organized in '{CLEAN_DATA_DIR}'.")

def train_model():
    print("\n--- Step 3: Starting Model Training ---")
    
    TRAIN_DIR = os.path.join(CLEAN_DATA_DIR, 'TRAIN')
    VAL_DIR = os.path.join(CLEAN_DATA_DIR, 'TEST')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    # Build Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]

    model.fit(train_generator, epochs=15, validation_data=val_generator, callbacks=callbacks)
    print(f"\n✅ Training Complete. Saved to {MODEL_FILE}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        setup_kaggle_api()
        download_dataset()
        organize_and_balance_data()
        train_model()
    else:
        print(f"✅ {MODEL_FILE} exists. Ready for app.py")