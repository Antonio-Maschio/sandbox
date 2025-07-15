import napari
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv("predictions.csv")
    print(data.head())
    print(data.columns)
    print(data.dtypes)

    tracks = data[["true_particle_id", "frame", "z", "y", "x"]].values
    track_features = (
        data[["mass", "event_label", "predicted_class", "correct_prediction"]]
        .astype(float)
        .to_dict(orient="list")
    )

    points = data[["frame", "z", "y", "x"]].values
    points_features = {
        "mass": data["mass"].astype(float).values,
        "event_label": data["event_label"].astype(str).values,
        "predicted_class": data["predicted_class"].astype(str).values,
        "correct_prediction": data["correct_prediction"].astype(bool).values,
    }

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_tracks(
        tracks,
        name="tracks",
        features=track_features,
    )
    viewer.add_points(
        points,
        name="points",
        features=points_features,
        size=1,
        face_color="predicted_class",
        border_color="correct_prediction",
        border_width=1,
        border_color_cycle=["white", "blue", "red", "orange", "darkblue"],
        face_color_cycle=["white", "blue", "red", "orange", "darkblue"],
    )
    napari.run()
