import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import model_training as mt

def initialize_model_output_shape(model, input_shape):
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    _ = model.predict(dummy_input)

def extract_features(img_url, model):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0'
    }

    try:
        response = requests.get(img_url, headers=headers)
        response.raise_for_status()
        if "image" not in response.headers["Content-Type"]:
            print(f"URL does not point to an image: {img_url}")
            return np.zeros((model.output_shape[-1],))

        img = Image.open(BytesIO(response.content)).convert('RGB').resize((32, 32))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        img_preprocessed = preprocess_input(img_array)

        features = model.predict(img_preprocessed)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image from URL {img_url}: {e}")
    return np.zeros((model.output_shape[-1],))

csv_file = 'inditextech_hackupc_challenge_images.csv'
test_df = pd.read_csv(csv_file)

url_columns = ['IMAGE_VERSION_1', 'IMAGE_VERSION_2', 'IMAGE_VERSION_3']
test_df_cleaned = test_df.dropna(subset=url_columns)

image_paths = test_df_cleaned['IMAGE_VERSION_1'].tolist() + test_df_cleaned['IMAGE_VERSION_2'].tolist() + test_df_cleaned['IMAGE_VERSION_3'].tolist()

cnn_model = mt.train_model()

initialize_model_output_shape(cnn_model, (32, 32, 3))

features = np.array([extract_features(path, cnn_model) for path in image_paths])

max_components = min(len(features), features.shape[1])

pca = PCA(n_components=max(1, max_components // 2))
features_reduced = pca.fit_transform(features)

num_clusters = max(1, len(image_paths) // 3)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(features_reduced)

cluster_labels = kmeans.labels_

results = pd.DataFrame({
    'Image Path': image_paths,
    'Cluster Label': cluster_labels
})

results_sorted = results.sort_values(by='Cluster Label')

output_csv = 'output_sorted_by_clusters.csv'
try:
    results_sorted.to_csv(output_csv, index=False)
    print(f"Results successfully written to {output_csv}")
except PermissionError as e:
    print(f"PermissionError: Unable to write to the file: {e}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
