import pandas as pd
import numpy as np
from collections import defaultdict
from enum import IntEnum
import sys

class EventLabel(IntEnum):
    NORMAL = 0
    MERGE = 1
    SPLIT = 2
    POST_MERGE = 3
    POST_SPLIT = 4

class ParticleDataValidator:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.errors = []
        self.warnings = []
        self.particles = {}
        self.events = {
            'merges': [],
            'splits': []
        }
        
        # Process data
        self._process_particles()
        self._detect_events()
    
    def log_error(self, message):
        self.errors.append(message)
        print(f"ERROR: {message}")
    
    def log_warning(self, message):
        self.warnings.append(message)
        print(f"WARNING: {message}")
    
    def _process_particles(self):
        """Process CSV data into particle trajectories"""
        for particle_id in self.df['particle'].unique():
            particle_data = self.df[self.df['particle'] == particle_id].sort_values('frame')
            
            self.particles[particle_id] = {
                'frames': particle_data['frame'].tolist(),
                'labels': particle_data['event_label'].tolist(),
                'parent_ids': particle_data['parent_ids'].iloc[0] if not pd.isna(particle_data['parent_ids'].iloc[0]) else '',
                'birth_frame': particle_data['frame'].min(),
                'death_frame': particle_data['frame'].max(),
                'duration': particle_data['duration'].iloc[0]
            }
    
    def _detect_events(self):
        """Detect merge and split events from the data"""
        # Group particles by their parent relationships
        for particle_id, data in self.particles.items():
            parent_ids_str = data['parent_ids']
            
            if parent_ids_str and ',' in parent_ids_str:
                # This particle has multiple parents - likely from a merge
                parent_ids = [int(x) for x in parent_ids_str.split(',') if x.strip()]
                if len(parent_ids) >= 2:
                    self.events['merges'].append({
                        'child_id': particle_id,
                        'parent_ids': parent_ids,
                        'frame': data['birth_frame']
                    })
            
            # Find children of this particle (splits)
            children = []
            for other_id, other_data in self.particles.items():
                if other_data['parent_ids']:
                    parent_ids = [int(x) for x in other_data['parent_ids'].split(',') if x.strip()]
                    if particle_id in parent_ids:
                        children.append(other_id)
            
            if len(children) >= 2:
                # Find children born at the same time (indicating a split)
                children_by_birth = defaultdict(list)
                for child_id in children:
                    birth_frame = self.particles[child_id]['birth_frame']
                    children_by_birth[birth_frame].append(child_id)
                
                for birth_frame, child_list in children_by_birth.items():
                    if len(child_list) >= 2:
                        self.events['splits'].append({
                            'parent_id': particle_id,
                            'child_ids': child_list,
                            'frame': birth_frame
                        })
    
    def validate_label_sequences(self):
        """Validate event label sequences for each particle"""
        print("\n=== VALIDATING LABEL SEQUENCES ===")
        
        for particle_id, data in self.particles.items():
            frames = data['frames']
            labels = data['labels']
            
            # Check for invalid transitions
            for i in range(len(labels) - 1):
                current_label = labels[i]
                next_label = labels[i + 1]
                current_frame = frames[i]
                next_frame = frames[i + 1]
                
                # Invalid sequences
                if current_label == EventLabel.SPLIT and next_label == EventLabel.POST_MERGE:
                    self.log_error(f"Particle {particle_id}: SPLIT at frame {current_frame} followed by POST_MERGE at frame {next_frame}")
                
                if current_label == EventLabel.MERGE and next_label == EventLabel.POST_SPLIT:
                    self.log_error(f"Particle {particle_id}: MERGE at frame {current_frame} followed by POST_SPLIT at frame {next_frame}")
                
                # Check for consecutive event labels
                event_labels = [EventLabel.MERGE, EventLabel.SPLIT, EventLabel.POST_MERGE, EventLabel.POST_SPLIT]
                if current_label in event_labels and next_label in event_labels:
                    self.log_warning(f"Particle {particle_id}: Consecutive event labels {EventLabel(current_label).name}->{EventLabel(next_label).name} at frames {current_frame}-{next_frame}")
                
                # Check for gaps in frames
                if next_frame != current_frame + 1:
                    self.log_warning(f"Particle {particle_id}: Frame gap from {current_frame} to {next_frame}")
            
            # Check for orphaned POST labels (POST_MERGE/POST_SPLIT without preceding MERGE/SPLIT)
            for i, label in enumerate(labels):
                frame = frames[i]
                
                if label == EventLabel.POST_MERGE:
                    # Look for preceding MERGE in this particle or related particles
                    found_merge = False
                    
                    # Check if this particle had a MERGE label recently
                    for j in range(max(0, i-5), i):
                        if labels[j] == EventLabel.MERGE:
                            found_merge = True
                            break
                    
                    if not found_merge:
                        # Check if this particle was involved in a merge event
                        for merge in self.events['merges']:
                            if merge['child_id'] == particle_id and abs(merge['frame'] - frame) <= 1:
                                found_merge = True
                                break
                    
                    if not found_merge:
                        self.log_error(f"Particle {particle_id}: POST_MERGE at frame {frame} without preceding MERGE event")
                
                if label == EventLabel.POST_SPLIT:
                    # Look for preceding SPLIT or involvement in split event
                    found_split = False
                    
                    # Check if this particle had a SPLIT label recently
                    for j in range(max(0, i-5), i):
                        if labels[j] == EventLabel.SPLIT:
                            found_split = True
                            break
                    
                    if not found_split:
                        # Check if this particle was involved in a split event
                        for split in self.events['splits']:
                            if (split['parent_id'] == particle_id or particle_id in split['child_ids']) and abs(split['frame'] - frame) <= 1:
                                found_split = True
                                break
                    
                    if not found_split:
                        self.log_error(f"Particle {particle_id}: POST_SPLIT at frame {frame} without preceding SPLIT event")
    
    def validate_merge_events(self):
        """Validate merge event labeling"""
        print("\n=== VALIDATING MERGE EVENTS ===")
        
        for merge in self.events['merges']:
            child_id = merge['child_id']
            parent_ids = merge['parent_ids']
            merge_frame = merge['frame']
            
            print(f"Checking merge: parents {parent_ids} -> child {child_id} at frame {merge_frame}")
            
            # Check if child particle has POST_MERGE label at birth
            child_data = self.particles[child_id]
            if child_data['birth_frame'] == merge_frame:
                birth_label = child_data['labels'][0] if child_data['labels'] else None
                if birth_label != EventLabel.POST_MERGE:
                    self.log_error(f"Merge child {child_id} should have POST_MERGE label at birth (frame {merge_frame}), got {EventLabel(birth_label).name if birth_label is not None else 'None'}")
            
            # Check if parent particles have MERGE labels before dying
            for parent_id in parent_ids:
                if parent_id in self.particles:
                    parent_data = self.particles[parent_id]
                    death_frame = parent_data['death_frame']
                    
                    # Check if parent died at the right time
                    if death_frame != merge_frame - 1 and death_frame != merge_frame:
                        self.log_warning(f"Merge parent {parent_id} death frame {death_frame} doesn't align with merge frame {merge_frame}")
                    
                    # Check if parent has MERGE label at death
                    if parent_data['labels']:
                        death_label = parent_data['labels'][-1]
                        if death_label != EventLabel.MERGE:
                            self.log_error(f"Merge parent {parent_id} should have MERGE label at death (frame {death_frame}), got {EventLabel(death_label).name}")
                else:
                    self.log_error(f"Merge parent {parent_id} not found in particle data")
    
    def validate_split_events(self):
        """Validate split event labeling"""
        print("\n=== VALIDATING SPLIT EVENTS ===")
        
        for split in self.events['splits']:
            parent_id = split['parent_id']
            child_ids = split['child_ids']
            split_frame = split['frame']
            
            print(f"Checking split: parent {parent_id} -> children {child_ids} at frame {split_frame}")
            
            # Check if parent particle has SPLIT label before dying/continuing
            if parent_id in self.particles:
                parent_data = self.particles[parent_id]
                death_frame = parent_data['death_frame']
                
                # Case 1: Normal split (parent dies)
                if death_frame == split_frame - 1 or death_frame == split_frame:
                    # Parent should have SPLIT label at death
                    if parent_data['labels']:
                        death_label = parent_data['labels'][-1]
                        if death_label != EventLabel.SPLIT:
                            self.log_error(f"Split parent {parent_id} should have SPLIT label at death (frame {death_frame}), got {EventLabel(death_label).name}")
                
                # Case 2: Linking split (parent continues)
                elif parent_id in child_ids:
                    # Parent should have POST_SPLIT label at split frame
                    frame_index = None
                    for i, frame in enumerate(parent_data['frames']):
                        if frame == split_frame:
                            frame_index = i
                            break
                    
                    if frame_index is not None and frame_index < len(parent_data['labels']):
                        split_label = parent_data['labels'][frame_index]
                        if split_label != EventLabel.POST_SPLIT:
                            self.log_error(f"Linking split parent {parent_id} should have POST_SPLIT label at frame {split_frame}, got {EventLabel(split_label).name}")
            
            # Check if child particles have POST_SPLIT labels at birth
            for child_id in child_ids:
                if child_id in self.particles:
                    child_data = self.particles[child_id]
                    if child_data['birth_frame'] == split_frame:
                        birth_label = child_data['labels'][0] if child_data['labels'] else None
                        if birth_label != EventLabel.POST_SPLIT:
                            self.log_error(f"Split child {child_id} should have POST_SPLIT label at birth (frame {split_frame}), got {EventLabel(birth_label).name if birth_label is not None else 'None'}")
                else:
                    self.log_error(f"Split child {child_id} not found in particle data")
    
    def validate_frame_consistency(self):
        """Validate frame numbering consistency"""
        print("\n=== VALIDATING FRAME CONSISTENCY ===")
        
        for particle_id, data in self.particles.items():
            frames = data['frames']
            birth_frame = data['birth_frame']
            duration = data['duration']
            
            # Check if frames are consecutive
            expected_frames = list(range(birth_frame, birth_frame + len(frames)))
            if frames != expected_frames:
                self.log_error(f"Particle {particle_id}: Non-consecutive frames. Expected {expected_frames[:5]}..., got {frames[:5]}...")
            
            # Check if duration matches actual frame count
            if len(frames) != duration:
                self.log_error(f"Particle {particle_id}: Duration {duration} doesn't match frame count {len(frames)}")
    
    def print_statistics(self):
        """Print validation statistics"""
        print("\n=== VALIDATION STATISTICS ===")
        print(f"Total particles: {len(self.particles)}")
        print(f"Total merge events detected: {len(self.events['merges'])}")
        print(f"Total split events detected: {len(self.events['splits'])}")
        print(f"Total errors: {len(self.errors)}")
        print(f"Total warnings: {len(self.warnings)}")
        
        # Label distribution
        all_labels = []
        for data in self.particles.values():
            all_labels.extend(data['labels'])
        
        label_counts = {}
        for label in all_labels:
            label_name = EventLabel(label).name
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("\nLabel distribution:")
        for label_name, count in sorted(label_counts.items()):
            print(f"  {label_name}: {count}")
    
    def validate_all(self):
        """Run all validation checks"""
        print("Starting validation of particle simulation data...")
        
        self.validate_frame_consistency()
        self.validate_label_sequences()
        self.validate_merge_events()
        self.validate_split_events()
        self.print_statistics()
        
        print(f"\n=== VALIDATION COMPLETE ===")
        print(f"Found {len(self.errors)} errors and {len(self.warnings)} warnings")
        
        if self.errors:
            print("\nFirst 10 errors:")
            for i, error in enumerate(self.errors[:10]):
                print(f"  {i+1}. {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        return len(self.errors) == 0

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_csv.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        validator = ParticleDataValidator(csv_file)
        is_valid = validator.validate_all()
        
        if is_valid:
            print("\n✅ Validation PASSED - No errors found!")
        else:
            print("\n❌ Validation FAILED - Errors found!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()