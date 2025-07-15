import pandas as pd
import numpy as np
from collections import defaultdict


def calculate_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def find_disappearing_particles(curr_frame, next_frame):
    curr_ids = set(curr_frame['particle'])
    next_ids = set(next_frame['particle'])
    return curr_ids - next_ids


def find_appearing_particles(curr_frame, next_frame):
    curr_ids = set(curr_frame['particle'])
    next_ids = set(next_frame['particle'])
    return next_ids - curr_ids


def process_split_events(df, distance_threshold=15):
    relationships = []
    validated_events = []
    
    for frame in range(int(df['frame'].min()), int(df['frame'].max())):
        curr_frame = df[df['frame'] == frame]
        next_frame = df[df['frame'] == frame + 1]
        
        if curr_frame.empty or next_frame.empty:
            continue
        
        split_particles = curr_frame[curr_frame['predicted_class'] == 2]
        post_split_particles = next_frame[next_frame['predicted_class'] == 4]
        
        disappearing = find_disappearing_particles(curr_frame, next_frame)
        appearing = find_appearing_particles(curr_frame, next_frame)
        
        for _, parent in split_particles.iterrows():
            if parent['particle'] not in disappearing:
                continue
            
            children_candidates = post_split_particles[
                post_split_particles['particle'].isin(appearing)
            ]
            
            matched_children = []
            for _, child in children_candidates.iterrows():
                dist = calculate_distance(
                    parent['x'], parent['y'], parent['z'],
                    child['x'], child['y'], child['z']
                )
                if dist <= distance_threshold:
                    matched_children.append(child)
                    relationships.append((child['particle'], parent['particle']))
            
            if len(matched_children) >= 2:
                validated_events.append({
                    'type': 'split',
                    'parent': parent['particle'],
                    'children': [c['particle'] for c in matched_children],
                    'frame': frame
                })
    
    return relationships, validated_events


def process_merge_events(df, distance_threshold=15):
    relationships = []
    validated_events = []
    
    for frame in range(int(df['frame'].min()), int(df['frame'].max())):
        curr_frame = df[df['frame'] == frame]
        next_frame = df[df['frame'] == frame + 1]
        
        if curr_frame.empty or next_frame.empty:
            continue
        
        merge_particles = curr_frame[curr_frame['predicted_class'] == 1]
        post_merge_particles = next_frame[next_frame['predicted_class'] == 3]
        
        disappearing = find_disappearing_particles(curr_frame, next_frame)
        appearing = find_appearing_particles(curr_frame, next_frame)
        
        merge_candidates = merge_particles[merge_particles['particle'].isin(disappearing)]
        
        if len(merge_candidates) < 2:
            continue
        
        centroid_x = merge_candidates['x'].mean()
        centroid_y = merge_candidates['y'].mean()
        centroid_z = merge_candidates['z'].mean()
        
        offspring_candidates = post_merge_particles[
            post_merge_particles['particle'].isin(appearing)
        ]
        
        best_offspring = None
        best_distance = float('inf')
        
        for _, offspring in offspring_candidates.iterrows():
            dist = calculate_distance(
                centroid_x, centroid_y, centroid_z,
                offspring['x'], offspring['y'], offspring['z']
            )
            if dist < best_distance and dist <= distance_threshold:
                best_distance = dist
                best_offspring = offspring
        
        if best_offspring is not None:
            for _, parent in merge_candidates.iterrows():
                relationships.append((best_offspring['particle'], parent['particle']))
            
            validated_events.append({
                'type': 'merge',
                'parents': list(merge_candidates['particle']),
                'child': best_offspring['particle'],
                'frame': frame
            })
    
    return relationships, validated_events


def process_continuations(df):
    relationships = []
    
    for frame in range(int(df['frame'].min()), int(df['frame'].max())):
        curr_frame = df[df['frame'] == frame]
        next_frame = df[df['frame'] == frame + 1]
        
        if curr_frame.empty or next_frame.empty:
            continue
        
        curr_non_events = curr_frame[curr_frame['predicted_class'] == 0]
        next_non_events = next_frame[next_frame['predicted_class'] == 0]
        
        continuing_ids = set(curr_non_events['particle']) & set(next_non_events['particle'])
        
        for particle_id in continuing_ids:
            relationships.append((particle_id, particle_id))
    
    return relationships


def build_full_lineage(all_relationships):
    direct_parents = defaultdict(list)
    for child_id, parent_id in all_relationships:
        if child_id != parent_id:
            direct_parents[child_id].append(parent_id)
    
    full_lineage = {}
    
    def get_ancestors(particle_id, visited=None):
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
            parent_ancestors = get_ancestors(parent_id, visited.copy())
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
    
    return full_lineage


def create_final_dataframe(df, all_relationships, events_summary):
    result = df.copy()
    
    full_lineage = build_full_lineage(all_relationships)
    
    result['parent_ids'] = result['particle'].apply(
        lambda x: full_lineage.get(x, [])
    )
    result['parent_ids'] = result['parent_ids'].apply(
        lambda x: str(x) if x else "[]"
    )
    
    result['event_info'] = ""
    for event in events_summary:
        if event['type'] == 'split':
            mask = result['particle'] == event['parent']
            result.loc[mask, 'event_info'] = f"Split parent → {event['children']}"
            
            for child in event['children']:
                mask = result['particle'] == child
                result.loc[mask, 'event_info'] = f"Split child from {event['parent']}"
        
        elif event['type'] == 'merge':
            for parent in event['parents']:
                mask = result['particle'] == parent
                result.loc[mask, 'event_info'] = f"Merge parent → {event['child']}"
            
            mask = result['particle'] == event['child']
            result.loc[mask, 'event_info'] = f"Merge child from {event['parents']}"
    
    return result


def validate_reconstruction(df, events):
    validation_report = {
        'total_particles': df['particle'].nunique(),
        'total_frames': df['frame'].nunique(),
        'split_events': len([e for e in events if e['type'] == 'split']),
        'merge_events': len([e for e in events if e['type'] == 'merge']),
        'orphan_post_splits': 0,
        'orphan_post_merges': 0,
        'unmatched_splits': 0,
        'unmatched_merges': 0
    }
    
    post_splits = df[df['predicted_class'] == 4]
    post_merges = df[df['predicted_class'] == 3]
    splits = df[df['predicted_class'] == 2]
    merges = df[df['predicted_class'] == 1]
    
    matched_post_splits = set()
    matched_post_merges = set()
    matched_splits = set()
    matched_merges = set()
    
    for event in events:
        if event['type'] == 'split':
            matched_splits.add(event['parent'])
            matched_post_splits.update(event['children'])
        elif event['type'] == 'merge':
            matched_merges.update(event['parents'])
            matched_post_merges.add(event['child'])
    
    validation_report['orphan_post_splits'] = len(
        set(post_splits['particle']) - matched_post_splits
    )
    validation_report['orphan_post_merges'] = len(
        set(post_merges['particle']) - matched_post_merges
    )
    validation_report['unmatched_splits'] = len(
        set(splits['particle']) - matched_splits
    )
    validation_report['unmatched_merges'] = len(
        set(merges['particle']) - matched_merges
    )
    
    return validation_report


def reconstruct_tracks_5class(input_file, output_file="reconstructed_tracks_5class.csv", 
                             distance_threshold=10):
    df = pd.read_csv(input_file)
    
    if 'frame' not in df.columns:
        raise ValueError("DataFrame must contain 'frame' column")
    
    if 'predicted_class' not in df.columns:
        raise ValueError("DataFrame must contain 'predicted_class' column")
    
    print("Processing split events...")
    split_relations, split_events = process_split_events(df, distance_threshold)
    
    print("Processing merge events...")
    merge_relations, merge_events = process_merge_events(df, distance_threshold)
    
    print("Processing continuations...")
    continuation_relations = process_continuations(df)
    
    all_relationships = split_relations + merge_relations + continuation_relations
    all_events = split_events + merge_events
    
    result = create_final_dataframe(df, all_relationships, all_events)
    result.to_csv(output_file, index=False)
    
    validation = validate_reconstruction(df, all_events)
    
    print(f"\nReconstruction Summary:")
    print(f"  Validated split events: {validation['split_events']}")
    print(f"  Validated merge events: {validation['merge_events']}")
    print(f"  Total relationships: {len(all_relationships)}")
    print(f"\nUnmatched predictions:")
    print(f"  Orphan post-splits: {validation['orphan_post_splits']}")
    print(f"  Orphan post-merges: {validation['orphan_post_merges']}")
    print(f"  Unmatched splits: {validation['unmatched_splits']}")
    print(f"  Unmatched merges: {validation['unmatched_merges']}")
    print(f"\nSaved to: {output_file}")
    
    return result, validation


if __name__ == "__main__":
    reconstruct_tracks_5class("predictions.csv")