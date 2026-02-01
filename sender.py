import os
import base64
import requests
import shutil
import pandas as pd
from datetime import datetime

#  Config
INPUT_FOLDER = './input_images'
ARCHIVE_FOLDER = './archive'
API_URL = 'http://localhost:8000/predict'
BATCH_SIZE = 10
THRESHOLD = 0.30

# Fashion-MNIST Class Labels
CLASS_LABELS = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def process_and_get_df():
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_results_data = []
    request_count = 0

    for i in range(0, len(all_files), BATCH_SIZE):
        batch_files = all_files[i : i + BATCH_SIZE]
        payload_batch = []

        for filename in batch_files:
            file_path = os.path.join(INPUT_FOLDER, filename)
            payload_batch.append({
                "id": filename,
                "data": encode_image(file_path)
            })

        try:
            response = requests.post(API_URL, json={"batch": payload_batch})
            request_count += 1
            print('Sending Batch..', request_count)
            response.raise_for_status()
            batch_results = response.json().get('results', [])

            for item in batch_results:
                img_id = item['id']
                probs = item['probabilities']
                
                # Identify max probability and its index
                max_prob = max(probs)
                pred_index = probs.index(max_prob)
                
                # Threshold Logic
                if max_prob < THRESHOLD:
                    pred_class = 'Review'
                else:
                    pred_class = CLASS_LABELS[pred_index]
                
                # Move to Archive
                src_path = os.path.join(INPUT_FOLDER, img_id)
                dest_path = os.path.join(ARCHIVE_FOLDER, img_id)
                if os.path.exists(src_path):
                    shutil.move(src_path, dest_path)

                all_results_data.append({
                    "Timestamp": datetime.now(),
                    "Image_ID": img_id,
                    "Predicted_Class": pred_class,
                    "Confidence": round(max_prob, 4),
                    "Raw_Probabilities": probs,
                    "Archive_Location": dest_path
                })

        except Exception as e:
            print(f"Error in batch {i}: {e}")

    return pd.DataFrame(all_results_data), request_count

if __name__ == "__main__":
    df, req = process_and_get_df()
    
    if not df.empty:
        # Summary of the results
        print("\n--- Processing Summary ---")
        print("\n Number of batches ran:", req)
        print(df['Predicted_Class'].value_counts())
        
        # Save to dataframe
        df.to_csv("fashion_results.csv", index=False)
    else:
        print("No images found to process.")