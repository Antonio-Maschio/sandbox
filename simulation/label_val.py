import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import warnings

class LabelVerifier:
    def __init__(self, df: pd.DataFrame, temporal_window: int = 0, verbose: bool = True):
        """
        Initialize the label verifier
        
        Args:
            df: DataFrame with simulation data
            temporal_window: Number of frames after event for post-event labeling
            verbose: Whether to print detailed information
        """
        self.df = df.copy()
        self.temporal_window = temporal_window
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        
    def parse_parent_ids(self, parent_str: str) -> List[int]:
        """Parse parent_ids string into list of integers"""
        if pd.isna(parent_str) or parent_str == '':
            return []
        return [int(x) for x in parent_str.split(',')]
    
    def find_merge_events(self) -> Dict[int, List[Dict]]:
        """
        Find all merge events by analyzing parent_ids
        Returns: {frame: [merge_event_info]}
        """
        merge_events = defaultdict(list)
        
        # Group by particle to analyze lineage
        particle_groups = self.df.groupby('particle')
        
        for particle_id, group in particle_groups:
            if group.empty:
                continue
                
            # Get first row to check parent_ids
            first_row = group.iloc[0]
            parent_ids = self.parse_parent_ids(first_row['parent_ids'])
            
            # If particle has multiple parents, it's a merge result
            if len(parent_ids) >= 2:
                birth_frame = int(first_row['frame'])
                
                # Find if this is a linking merge (one parent continues)
                is_linking = particle_id in parent_ids
                
                merge_events[birth_frame].append({
                    'child_id': particle_id,
                    'parent_ids': parent_ids,
                    'is_linking': is_linking,
                    'type': 'merge'
                })
        
        return merge_events
    
    def find_split_events(self) -> Dict[int, List[Dict]]:
        """
        Find all split events by analyzing parent_ids
        Returns: {frame: [split_event_info]}
        """
        split_events = defaultdict(list)
        
        # Group by frame to find particles born in same frame with same parent
        frame_groups = self.df.groupby('frame')
        
        for frame, group in frame_groups:
            # Get unique particles in this frame
            particles_in_frame = group.groupby('particle').first()
            
            # Group by parent_ids to find splits
            parent_groups = defaultdict(list)
            for particle_id, row in particles_in_frame.iterrows():
                parent_ids = self.parse_parent_ids(row['parent_ids'])
                if len(parent_ids) == 1:  # Single parent means potential split
                    parent_groups[parent_ids[0]].append(particle_id)
            
            # Find splits (multiple children from same parent)
            for parent_id, children in parent_groups.items():
                if len(children) >= 2:
                    # Check if this is a linking split (parent continues)
                    is_linking = parent_id in children
                    
                    split_events[frame].append({
                        'parent_id': parent_id,
                        'child_ids': children,
                        'is_linking': is_linking,
                        'type': 'split'
                    })
        
        return split_events
    
    def verify_merge_labels(self, merge_events: Dict[int, List[Dict]]) -> bool:
        """Verify merge event labels are correct"""
        all_correct = True
        
        for frame, events in merge_events.items():
            for event in events:
                parent_ids = event['parent_ids']
                child_id = event['child_id']
                is_linking = event['is_linking']
                
                # Check parent particles have MERGE label in their last frame
                for parent_id in parent_ids:
                    if parent_id == child_id and is_linking:
                        # For linking merges, continuing particle should have MERGE then POST_MERGE
                        continue
                    
                    # Find last occurrence of parent particle
                    parent_data = self.df[self.df['particle'] == parent_id]
                    if parent_data.empty:
                        self.errors.append(f"Frame {frame}: Parent particle {parent_id} not found in data")
                        all_correct = False
                        continue
                    
                    last_parent_row = parent_data.iloc[-1]
                    if last_parent_row['event_label'] != 1:  # MERGE = 1
                        self.errors.append(
                            f"Frame {frame}: Parent particle {parent_id} should have MERGE label (1), "
                            f"but has {last_parent_row['event_label']}"
                        )
                        all_correct = False
                
                # Check child particle has POST_MERGE label initially
                child_data = self.df[self.df['particle'] == child_id]
                if child_data.empty:
                    self.errors.append(f"Frame {frame}: Child particle {child_id} not found in data")
                    all_correct = False
                    continue
                
                first_child_row = child_data.iloc[0]
                expected_frames_with_post_merge = min(self.temporal_window + 1, len(child_data))
                
                if is_linking:
                    # For linking merges, check sequence: MERGE -> POST_MERGE
                    if len(child_data) >= 2:
                        # Check that particle has MERGE label before POST_MERGE
                        merge_found = False
                        for i, row in child_data.iterrows():
                            if row['event_label'] == 1:  # MERGE
                                merge_found = True
                            elif row['event_label'] == 3 and merge_found:  # POST_MERGE after MERGE
                                break
                        
                        if not merge_found:
                            self.errors.append(
                                f"Frame {frame}: Linking merge child {child_id} should have MERGE label before POST_MERGE"
                            )
                            all_correct = False
                else:
                    # For normal merges, child should start with POST_MERGE
                    if first_child_row['event_label'] != 3:  # POST_MERGE = 3
                        self.errors.append(
                            f"Frame {frame}: Normal merge child {child_id} should start with POST_MERGE label (3), "
                            f"but has {first_child_row['event_label']}"
                        )
                        all_correct = False
        
        return all_correct
    
    def verify_split_labels(self, split_events: Dict[int, List[Dict]]) -> bool:
        """Verify split event labels are correct"""
        all_correct = True
        
        for frame, events in split_events.items():
            for event in events:
                parent_id = event['parent_id']
                child_ids = event['child_ids']
                is_linking = event['is_linking']
                
                # Check parent particle has SPLIT label in its last frame (if not linking)
                if not is_linking:
                    parent_data = self.df[self.df['particle'] == parent_id]
                    if parent_data.empty:
                        self.errors.append(f"Frame {frame}: Parent particle {parent_id} not found in data")
                        all_correct = False
                        continue
                    
                    last_parent_row = parent_data.iloc[-1]
                    if last_parent_row['event_label'] != 2:  # SPLIT = 2
                        self.errors.append(
                            f"Frame {frame}: Parent particle {parent_id} should have SPLIT label (2), "
                            f"but has {last_parent_row['event_label']}"
                        )
                        all_correct = False
                
                # Check child particles have POST_SPLIT label initially
                for child_id in child_ids:
                    child_data = self.df[self.df['particle'] == child_id]
                    if child_data.empty:
                        self.errors.append(f"Frame {frame}: Child particle {child_id} not found in data")
                        all_correct = False
                        continue
                    
                    if is_linking and child_id == parent_id:
                        # For linking splits, continuing particle should have SPLIT -> POST_SPLIT
                        split_found = False
                        post_split_found = False
                        for i, row in child_data.iterrows():
                            if row['event_label'] == 2:  # SPLIT
                                split_found = True
                            elif row['event_label'] == 4 and split_found:  # POST_SPLIT after SPLIT
                                post_split_found = True
                                break
                        
                        if not split_found:
                            self.errors.append(
                                f"Frame {frame}: Linking split continuing particle {child_id} should have SPLIT label"
                            )
                            all_correct = False
                        if not post_split_found and len(child_data) > 1:
                            self.errors.append(
                                f"Frame {frame}: Linking split continuing particle {child_id} should have POST_SPLIT after SPLIT"
                            )
                            all_correct = False
                    else:
                        # For normal splits or new particles in linking splits
                        first_child_row = child_data.iloc[0]
                        if first_child_row['event_label'] != 4:  # POST_SPLIT = 4
                            self.errors.append(
                                f"Frame {frame}: Split child {child_id} should start with POST_SPLIT label (4), "
                                f"but has {first_child_row['event_label']}"
                            )
                            all_correct = False
        
        return all_correct
    
    def check_label_ratios(self) -> Dict[str, float]:
        """Check the ratios between different event labels"""
        label_counts = self.df['event_label'].value_counts()
        
        ratios = {}
        total_events = label_counts.get(1, 0) + label_counts.get(2, 0)  # MERGE + SPLIT
        total_post_events = label_counts.get(3, 0) + label_counts.get(4, 0)  # POST_MERGE + POST_SPLIT
        
        if total_events > 0:
            ratios['post_events_to_events'] = total_post_events / total_events
        
        if label_counts.get(1, 0) > 0:
            ratios['post_merge_to_merge'] = label_counts.get(3, 0) / label_counts.get(1, 0)
        
        if label_counts.get(2, 0) > 0:
            ratios['post_split_to_split'] = label_counts.get(4, 0) / label_counts.get(2, 0)
        
        return ratios
    
    def verify_labels(self) -> Dict[str, any]:
        """Main verification function"""
        if self.verbose:
            print("Starting label verification...")
        
        # Find events
        merge_events = self.find_merge_events()
        split_events = self.find_split_events()
        
        if self.verbose:
            print(f"Found {sum(len(events) for events in merge_events.values())} merge events")
            print(f"Found {sum(len(events) for events in split_events.values())} split events")
        
        # Verify labels
        merge_correct = self.verify_merge_labels(merge_events)
        split_correct = self.verify_split_labels(split_events)
        
        # Check ratios
        ratios = self.check_label_ratios()
        
        # Summary
        label_counts = self.df['event_label'].value_counts().sort_index()
        
        results = {
            'merge_labels_correct': merge_correct,
            'split_labels_correct': split_correct,
            'overall_correct': merge_correct and split_correct and len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'label_counts': label_counts.to_dict(),
            'ratios': ratios,
            'merge_events': merge_events,
            'split_events': split_events
        }
        
        if self.verbose:
            self.print_results(results)
        
        return results
    
    def print_results(self, results: Dict):
        """Print verification results"""
        print("\n" + "="*50)
        print("LABEL VERIFICATION RESULTS")
        print("="*50)
        
        print(f"Overall Correct: {results['overall_correct']}")
        print(f"Merge Labels Correct: {results['merge_labels_correct']}")
        print(f"Split Labels Correct: {results['split_labels_correct']}")
        
        print(f"\nLabel Counts:")
        for label, count in results['label_counts'].items():
            label_names = {0: 'NORMAL', 1: 'MERGE', 2: 'SPLIT', 3: 'POST_MERGE', 4: 'POST_SPLIT'}
            print(f"  {label_names.get(label, f'Unknown({label})')}: {count}")
        
        print(f"\nRatios:")
        for ratio_name, ratio_value in results['ratios'].items():
            print(f"  {ratio_name}: {ratio_value:.2f}")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings'][:5]:  # Show first 5 warnings
                print(f"  - {warning}")
            if len(results['warnings']) > 5:
                print(f"  ... and {len(results['warnings']) - 5} more warnings")


def verify_simulation_labels(csv_file: str, temporal_window: int = 0, verbose: bool = True) -> Dict[str, any]:
    """
    Convenience function to verify labels from a CSV file
    
    Args:
        csv_file: Path to the CSV file with simulation data
        temporal_window: Number of frames after event for post-event labeling
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with verification results
    """
    df = pd.read_csv(csv_file)
    verifier = LabelVerifier(df, temporal_window, verbose)
    return verifier.verify_labels()


# Example usage
if __name__ == "__main__":
    # Example of how to use the verification function
    # Replace with your actual CSV file path
    # csv_file = "data/tracked_simdata_partiallinking/tracked_particles_3d_00000.csv"
    csv_file ='data/tracked_simdata_clean/tracked_particles_3d_00000.csv'
    
    try:
        results = verify_simulation_labels(csv_file, temporal_window=0, verbose=True)
        
        if results['overall_correct']:
            print("\n✅ All labels are correctly assigned!")
        else:
            print(f"\n❌ Found {len(results['errors'])} labeling errors")
            
    except FileNotFoundError:
        print(f"File {csv_file} not found. Please run a simulation first.")
    except Exception as e:
        print(f"Error during verification: {e}")