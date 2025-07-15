import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_distance(p1, p2):
    return np.sqrt((p2['x'] - p1['x'])**2 + 
                   (p2['y'] - p1['y'])**2 + 
                   (p2['z'] - p1['z'])**2)


def find_k_nearest_neighbors(particle, candidates, k=3):
    distances = []
    for _, candidate in candidates.iterrows():
        dist = calculate_distance(particle, candidate)
        distances.append((dist, candidate['particle']))
    
    distances.sort()
    return [pid for dist, pid in distances[:k]]


def reconstruct_tracks_from_predictions(df, k_neighbors=3, max_distance=15):
    df = df.sort_values(['frame', 'particle']).reset_index(drop=True)
    
    particle_history = defaultdict(list)
    parent_relationships = []
    
    frames = sorted(df['frame'].unique())
    
    for i in range(len(frames) - 1):
        curr_frame = df[df['frame'] == frames[i]]
        next_frame = df[df['frame'] == frames[i + 1]]
        
        curr_particles = set(curr_frame['particle'])
        next_particles = set(next_frame['particle'])
        
        continuing_particles = curr_particles & next_particles
        disappearing_particles = curr_particles - next_particles
        new_particles = next_particles - curr_particles
        
        splits = curr_frame[curr_frame['predicted_class'] == 2]
        merges = curr_frame[curr_frame['predicted_class'] == 1]
        post_splits = next_frame[next_frame['predicted_class'] == 4]
        post_merges = next_frame[next_frame['predicted_class'] == 3]
        
        for _, split_parent in splits.iterrows():
            if split_parent['particle'] in disappearing_particles:
                candidates = next_frame[next_frame['particle'].isin(new_particles)]
                if not candidates.empty:
                    children = find_k_nearest_neighbors(split_parent, candidates, k=k_neighbors)
                    for child_id in children:
                        child = candidates[candidates['particle'] == child_id].iloc[0]
                        if calculate_distance(split_parent, child) <= max_distance:
                            parent_relationships.append((child_id, split_parent['particle']))
        
        if len(merges) >= 2:
            merge_group = merges[merges['particle'].isin(disappearing_particles)]
            if len(merge_group) >= 2:
                centroid = {
                    'x': merge_group['x'].mean(),
                    'y': merge_group['y'].mean(),
                    'z': merge_group['z'].mean()
                }
                
                candidates = next_frame[next_frame['particle'].isin(new_particles)]
                if not candidates.empty:
                    nearest = find_k_nearest_neighbors(centroid, candidates, k=1)
                    if nearest:
                        offspring_id = nearest[0]
                        offspring = candidates[candidates['particle'] == offspring_id].iloc[0]
                        if calculate_distance(centroid, offspring) <= max_distance:
                            for _, parent in merge_group.iterrows():
                                parent_relationships.append((offspring_id, parent['particle']))
        
        for _, particle in post_splits.iterrows():
            if particle['particle'] in new_particles:
                candidates = curr_frame[curr_frame['particle'].isin(disappearing_particles)]
                if not candidates.empty:
                    parents = find_k_nearest_neighbors(particle, candidates, k=1)
                    for parent_id in parents:
                        parent = candidates[candidates['particle'] == parent_id].iloc[0]
                        if calculate_distance(particle, parent) <= max_distance:
                            parent_relationships.append((particle['particle'], parent_id))
        
        for _, particle in post_merges.iterrows():
            if particle['particle'] in new_particles:
                candidates = curr_frame[curr_frame['particle'].isin(disappearing_particles)]
                if not candidates.empty:
                    parents = find_k_nearest_neighbors(particle, candidates, k=k_neighbors)
                    for parent_id in parents:
                        parent = candidates[candidates['particle'] == parent_id].iloc[0]
                        if calculate_distance(particle, parent) <= max_distance:
                            parent_relationships.append((particle['particle'], parent_id))
    
    return build_complete_lineage(df, parent_relationships)


def build_complete_lineage(df, relationships):
    direct_parents = defaultdict(list)
    for child_id, parent_id in relationships:
        direct_parents[child_id].append(parent_id)
    
    full_lineage = {}
    
    def get_all_ancestors(particle_id, visited=None):
        if visited is None:
            visited = set()
        
        if particle_id in visited:
            return []
        
        visited.add(particle_id)
        
        if particle_id in full_lineage:
            return full_lineage[particle_id].copy()
        
        if particle_id not in direct_parents:
            full_lineage[particle_id] = []
            return []
        
        ancestors = []
        for parent_id in direct_parents[particle_id]:
            parent_ancestors = get_all_ancestors(parent_id, visited)
            ancestors.extend(parent_ancestors)
            ancestors.append(parent_id)
        
        unique_ancestors = []
        seen = set()
        for ancestor in ancestors:
            if ancestor not in seen:
                unique_ancestors.append(ancestor)
                seen.add(ancestor)
        
        full_lineage[particle_id] = unique_ancestors
        return unique_ancestors
    
    all_particles = set(df['particle'].unique())
    for particle_id in all_particles:
        get_all_ancestors(particle_id)
    
    result = df.copy()
    result['parent_ids'] = result['particle'].apply(
        lambda x: full_lineage.get(x, [])
    )
    result['parent_ids_str'] = result['parent_ids'].apply(
        lambda x: str(x) if x else "[]"
    )
    
    return result, relationships


def track_reconstruction_pipeline(predictions_csv, output_csv="reconstructed_tracks.csv", 
                                 k_neighbors=3, max_distance=15):
    df = pd.read_csv(predictions_csv)
    
    result, relationships = reconstruct_tracks_from_predictions(
        df, k_neighbors=k_neighbors, max_distance=max_distance
    )
    
    result.to_csv(output_csv, index=False)
    
    splits_found = sum(1 for _, parent in relationships 
                      if df[df['particle'] == parent]['predicted_class'].iloc[0] == 2)
    merges_found = sum(1 for _, parent in relationships 
                      if df[df['particle'] == parent]['predicted_class'].iloc[0] == 1)
    
    print(f"Track reconstruction complete:")
    print(f"  Splits found: {splits_found}")
    print(f"  Merges found: {merges_found}")
    print(f"  Total relationships: {len(relationships)}")
    print(f"  Saved to: {output_csv}")
    
    return result


if __name__ == "__main__":
    track_reconstruction_pipeline(
        "predictions.csv",
        "reconstructed_tracks.csv",
        k_neighbors=3,
        max_distance=15
    )