import napari
import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv("data/tracked_simdata_partiallinking/tracked_particles_3d_00000.csv")
    # data = pd.read_csv("data/3class/tracked_simdata_3d_temporal_Highmass/tracked_particles_3d_00000.csv")

    print(data.head())
    print(data.columns)
    print(data.dtypes)

    tracks = data[["true_particle_id", "frame", "z", "y", "x"]].values
    track_features = (
        data[["mass", "event_label"]]
        .astype(float)
        .to_dict(orient="list")
    )

    points = data[["frame", "z", "y", "x"]].values
    points_features = {
        "mass": data["mass"].astype(float).values,
        "event_label": data["event_label"].astype(str).values
        # "predicted_class": data["predicted_class"].astype(str).values,
        # "correct_prediction": data["correct_prediction"].astype(bool).values,
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
        face_color="event_label",  # Use event_label for face color
        border_color="event_label",  # Use event_label for border color too
        border_width=1,
        border_color_cycle=["white", "orange", "lime", "red", "green"],
        face_color_cycle=["white", "orange", "lime", "red", "green"],
    )
    napari.run()