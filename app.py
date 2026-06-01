import streamlit as st
import os
import zipfile
import shutil

import pandas as pd

from PIL import Image

from utils import predict_label

# =================================

st.set_page_config(
    page_title="CLIP Dataset Labeler",
    layout="wide"
)

st.title(
    "📸 CLIP Dataset Auto Labeling Tool"
)

# =================================

uploaded_zip = st.file_uploader(
    "Upload Dataset ZIP",
    type=["zip"]
)

# =================================

label_text = st.text_input(
    "Enter Classes (comma separated)",
    "cat,dog,horse,cow"
)

labels = [
    x.strip()
    for x in label_text.split(",")
]

# =================================

if uploaded_zip:

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    zip_path = os.path.join(
        "uploads",
        uploaded_zip.name
    )

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    extract_folder = "dataset"

    if os.path.exists(extract_folder):
        shutil.rmtree(extract_folder)

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            extract_folder
        )

    st.success(
        "Dataset Extracted Successfully"
    )

    results = []

    images = []

    for root, dirs, files in os.walk(
        extract_folder
    ):

        for file in files:

            if file.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                images.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    progress = st.progress(0)

    for i, image_path in enumerate(images):

        image = Image.open(
            image_path
        ).convert("RGB")

        label, confidence = predict_label(
            image,
            labels
        )

        os.makedirs(
            f"labeled_dataset/{label}",
            exist_ok=True
        )

        shutil.copy(
            image_path,
            f"labeled_dataset/{label}"
        )

        results.append(
            [
                os.path.basename(
                    image_path
                ),
                label,
                confidence
            ]
        )

        progress.progress(
            (i + 1) / len(images)
        )

    df = pd.DataFrame(
        results,
        columns=[
            "Image",
            "Label",
            "Confidence"
        ]
    )

    df.to_csv(
        "auto_labels.csv",
        index=False
    )

    st.success(
        "Labeling Completed"
    )

    st.dataframe(df)

    with open(
        "auto_labels.csv",
        "rb"
    ) as f:

        st.download_button(
            "Download CSV",
            f,
            "auto_labels.csv"
        )
