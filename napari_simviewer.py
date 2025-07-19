import napari
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # data = pd.read_csv("data/tracked_simdata_partiallinking/tracked_particles_3d_00000.csv")
    data = pd.read_csv("data/tracked_simdata_clean/tracked_particles_3d_00000.csv")

    print(data.head())
    print(data.columns)
    print(data.dtypes)

    
    # Create explicit color mapping for event labels
    event_color_map = {
        "0": "white",   # Background/untracked
        "1": "darkcyan",    # Event type 1
        "2": "coral",     # Event type 2
        "3": "navy",    # Event type 3
        "4": "red",   # Event type 4
        # Add more mappings as needed
    }

    # Convert event labels to strings for consistent mapping
    event_labels = data["event_label"].astype(str).values
    
    # Map colors based on event labels
    face_colors = [event_color_map.get(label, "gray") for label in event_labels]
    border_colors = face_colors  # Use same colors for borders

    tracks = data[["true_particle_id", "frame", "z", "y", "x"]].values
    track_features = data[["mass", "event_label"]].astype(float).to_dict(orient="list")

    points = data[["frame", "z", "y", "x"]].values
    points_features = {
        "mass": data["mass"].astype(float).values,
        "event_label": event_labels,
        "id": data["true_particle_id"].astype(str).values
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
        face_color=face_colors,  # Use event_label for face color
        border_color=border_colors,  # Use event_label for border color too
        border_width=1,
        # border_color_cycle=["white", "navy", "red", "blue", "green"],
        # face_color_cycle=["white", "navy", "red", "blue", "green"],
    )
    napari.run()