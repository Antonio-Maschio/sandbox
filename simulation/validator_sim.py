import pandas as pd
import numpy as np

def analyze_simulation_events(csv_file):
    """Analyze particle events from simulation CSV"""
    df = pd.read_csv(csv_file)
    
    print("=== PARTICLE EVENT ANALYSIS ===\n")
    
    # Key frames of interest
    frames_of_interest = [163, 164, 171, 172, 178]
    
    # Track particles involved in events
    event_particles = {
        24: "splits at frame 163",
        6: "splits at frame 164", 
        52: "merges at frame 171",
        72: "created from 24 split, merges at 171",
        60: "merges at frame 172",
        9: "splits at frame 178"
    }
    
    for particle_id, event_desc in event_particles.items():
        print(f"\n--- Particle {particle_id} ({event_desc}) ---")
        
        # Get all data for this particle
        particle_data = df[df['particle'] == particle_id].sort_values('frame')
        
        if not particle_data.empty:
            print(f"Birth frame: {particle_data['frame'].min()}")
            print(f"Death frame: {particle_data['frame'].max()}")
            print(f"Parent IDs: {particle_data['parent_ids'].iloc[0]}")
            
            # Show data around key frames
            for frame in frames_of_interest:
                frame_data = particle_data[particle_data['frame'] == frame]
                if not frame_data.empty:
                    row = frame_data.iloc[0]
                    print(f"  Frame {frame}: label={row['event_label']}, mass={row['mass']:.0f}")
    
    # Analyze new particles appearing in key frames
    print("\n\n=== NEW PARTICLES IN KEY FRAMES ===")
    
    for frame in frames_of_interest:
        frame_data = df[df['frame'] == frame]
        
        # Find particles that first appear in this frame
        new_particles = []
        for particle_id in frame_data['particle'].unique():
            particle_history = df[df['particle'] == particle_id]
            if particle_history['frame'].min() == frame:
                new_particles.append(particle_id)
        
        if new_particles:
            print(f"\nFrame {frame} - New particles: {new_particles}")
            for pid in new_particles:
                pdata = frame_data[frame_data['particle'] == pid].iloc[0]
                print(f"  Particle {pid}: parent_ids='{pdata['parent_ids']}', label={pdata['event_label']}")
    
    # Check for particles that disappear and reappear
    print("\n\n=== PARTICLE LIFECYCLE ISSUES ===")
    
    for particle_id in df['particle'].unique():
        particle_data = df[df['particle'] == particle_id].sort_values('frame')
        frames = particle_data['frame'].values
        
        # Check for gaps in frames
        if len(frames) > 1:
            expected_frames = np.arange(frames[0], frames[-1] + 1)
            missing_frames = set(expected_frames) - set(frames)
            
            if missing_frames:
                print(f"\nParticle {particle_id} has gaps in frames: {sorted(missing_frames)}")
    
    # Analyze merges and splits
    print("\n\n=== EVENT ANALYSIS ===")
    
    # Find all POST_MERGE (3) and POST_SPLIT (4) events
    post_merge = df[df['event_label'] == 3]
    post_split = df[df['event_label'] == 4]
    
    print(f"\nPOST_MERGE events:")
    for _, row in post_merge.iterrows():
        if row['frame'] in frames_of_interest:
            print(f"  Frame {row['frame']}: Particle {row['particle']} (parent_ids: {row['parent_ids']})")
    
    print(f"\nPOST_SPLIT events:")
    for _, row in post_split.iterrows():
        if row['frame'] in frames_of_interest:
            print(f"  Frame {row['frame']}: Particle {row['particle']} (parent_ids: {row['parent_ids']})")

# Usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    analyze_simulation_events("data/tracked_simdata_clean/tracked_particles_3d_00000.csv")