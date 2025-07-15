import napari
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv("predictions.csv")
    
    # Ensure predicted_class is treated as categorical for coloring
    data["predicted_class"] = data["predicted_class"].astype(str)

    # Prepare tracks and features
    tracks = data[["true_particle_id", "frame", "z", "y", "x"]].values
    track_features = (
        data[["mass", "event_label", "predicted_class", "correct_prediction"]]
        .astype(float)
        .to_dict(orient="list")
    )

    # Prepare points and features
    points = data[["frame", "z", "y", "x"]].values
    points_features = {
        "mass": data["mass"].astype(float).values,
        "event_label": data["event_label"].astype(str).values,
        "predicted_class": data["predicted_class"].values,  # already str
        "correct_prediction": data["correct_prediction"].astype(bool).values,
    }

    # Optional: Define your own color cycles
    # You can extract color codes from your Plotly visual if needed
    custom_face_colors = {
        "0": "white",
        "1": "blue",
        "2": "red",
        "3": "orange",
        "4": "darkblue",
    }

    # Map custom colors to predicted classes (if known)
    predicted_class_labels = sorted(data["predicted_class"].unique())
    face_color_cycle = [custom_face_colors.get(label, "gray") for label in predicted_class_labels]

    viewer = napari.Viewer(ndisplay=3)
    
    # Add tracks without edge color modifications
    viewer.add_tracks(
        tracks,
        name="tracks",
        features=track_features,
    )

    # Add colored points based on predicted class
    viewer.add_points(
        points,
        name="points",
        features=points_features,
        size=1,
        face_color="predicted_class",
        face_color_cycle=face_color_cycle,
        border_color="correct_prediction",
        border_width=1,
        # Optional: define border color cycle if `correct_prediction` is categorical
        border_color_cycle=["red", "green"],  # False = red, True = green
    )

    napari.run()
