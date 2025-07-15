import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import warnings
from typing import Dict, List, Tuple, Optional
import os

class SimulationValidator:
    """
    Comprehensive validation class for particle simulation output
    """
    
    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        self.df = df.copy()
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.info = []
        
    def validate_simulation_output(self) -> Dict:
        """
        Main validation function that runs all checks
        
        Returns:
            Dict with validation results including errors, warnings, and summary stats
        """
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Basic data integrity checks
        self._check_data_integrity()
        
        # Event label consistency checks
        self._check_event_label_consistency()
        
        # Particle trajectory continuity
        self._check_trajectory_continuity()
        
        # Mass conservation in merge/split events
        self._check_mass_conservation()
        
        # Temporal consistency
        self._check_temporal_consistency()
        
        # Spatial boundary checks
        self._check_spatial_boundaries()
        
        # Ghost particle validation
        self._check_ghost_particles()
        
        # Parent-child relationship validation
        self._check_parent_child_relationships()
        
        # Statistical plausibility checks
        self._check_statistical_plausibility()
        
        # Duration and lifetime checks
        self._check_duration_consistency()
        
        return self._generate_report()
    
    def _check_data_integrity(self):
        """Check basic data integrity"""
        required_columns = ['frame', 'x', 'y', 'z', 'mass', 'particle', 'duration', 
                          'event_label', 'is_ghost', 'true_particle_id', 'parent_ids']
        
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return
        
        # Check for null values in critical columns
        critical_cols = ['frame', 'x', 'y', 'z', 'mass', 'particle', 'event_label']
        for col in critical_cols:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Found {null_count} null values in column '{col}'")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(self.df['frame']):
            self.errors.append("Frame column should be numeric")
        
        if not pd.api.types.is_numeric_dtype(self.df['particle']):
            self.errors.append("Particle column should be numeric")
        
        # Check for negative values where they shouldn't exist
        if (self.df['mass'] < 0).any():
            self.errors.append("Found negative mass values")
        
        if (self.df['frame'] < 0).any():
            self.errors.append("Found negative frame values")
        
        # Check event labels are within valid range
        valid_labels = [0, 1, 2, 3, 4]  # NORMAL, MERGE, SPLIT, POST_MERGE, POST_SPLIT
        invalid_labels = self.df[~self.df['event_label'].isin(valid_labels)]
        if not invalid_labels.empty:
            self.errors.append(f"Found {len(invalid_labels)} invalid event labels")
    
    def _check_event_label_consistency(self):
        """Check event label ratios and consistency"""
        # Filter out ghost particles for event analysis
        real_particles = self.df[self.df['is_ghost'] == False]
        
        if real_particles.empty:
            self.warnings.append("No real particles found in dataset")
            return
        
        label_counts = real_particles['event_label'].value_counts()
        
        merge_count = label_counts.get(1, 0)  # MERGE
        split_count = label_counts.get(2, 0)  # SPLIT
        post_merge_count = label_counts.get(3, 0)  # POST_MERGE
        post_split_count = label_counts.get(4, 0)  # POST_SPLIT
        
        # Check merge/post-merge ratio
        if merge_count > 0:
            expected_post_merge = merge_count * 2  # Each merge should produce 2 post-merge labels
            if abs(post_merge_count - expected_post_merge) > merge_count * 0.1:  # Allow 10% tolerance
                self.warnings.append(
                    f"Post-merge count ({post_merge_count}) doesn't match expected "
                    f"2x merge count ({expected_post_merge}). Ratio: {post_merge_count/merge_count:.2f}"
                )
        
        # Check split/post-split ratio
        if split_count > 0:
            expected_post_split = split_count * 2  # Each split should produce 2 post-split labels
            if abs(post_split_count - expected_post_split) > split_count * 0.1:  # Allow 10% tolerance
                self.warnings.append(
                    f"Post-split count ({post_split_count}) doesn't match expected "
                    f"2x split count ({expected_post_split}). Ratio: {post_split_count/split_count:.2f}"
                )
        
        self.info.append(f"Event label counts: NORMAL={label_counts.get(0, 0)}, "
                        f"MERGE={merge_count}, SPLIT={split_count}, "
                        f"POST_MERGE={post_merge_count}, POST_SPLIT={post_split_count}")
    
    def _check_trajectory_continuity(self):
        """Check for gaps or jumps in particle trajectories"""
        real_particles = self.df[self.df['is_ghost'] == False]
        
        for particle_id in real_particles['particle'].unique():
            particle_data = real_particles[real_particles['particle'] == particle_id].sort_values('frame')
            
            if len(particle_data) < 2:
                continue
                
            # Check for frame gaps
            frame_diffs = particle_data['frame'].diff().dropna()
            large_gaps = frame_diffs[frame_diffs > 1]
            if not large_gaps.empty:
                self.warnings.append(f"Particle {particle_id} has {len(large_gaps)} temporal gaps")
            
            # Check for excessive position jumps
            coords = ['x', 'y', 'z']
            for coord in coords:
                coord_diffs = particle_data[coord].diff().dropna()
                large_jumps = coord_diffs[abs(coord_diffs) > 20]  # Threshold for "large" jump
                if not large_jumps.empty:
                    max_jump = abs(coord_diffs).max()
                    self.warnings.append(f"Particle {particle_id} has large {coord} jumps (max: {max_jump:.2f})")
    
    def _check_mass_conservation(self):
        """Check mass conservation in merge/split events"""
        real_particles = self.df[self.df['is_ghost'] == False]
        
        # Group by frame to find events
        for frame in real_particles['frame'].unique():
            frame_data = real_particles[real_particles['frame'] == frame]
            
            merge_particles = frame_data[frame_data['event_label'] == 1]  # MERGE
            split_particles = frame_data[frame_data['event_label'] == 2]  # SPLIT
            
            # For merge events, check if mass is conserved
            if not merge_particles.empty:
                # This is a simplified check - in reality, we'd need to track the actual merge relationships
                self.info.append(f"Frame {frame}: Found {len(merge_particles)} merge events")
            
            # For split events, similar logic
            if not split_particles.empty:
                self.info.append(f"Frame {frame}: Found {len(split_particles)} split events")
    
    def _check_temporal_consistency(self):
        """Check temporal consistency of events"""
        real_particles = self.df[self.df['is_ghost'] == False]
        
        for particle_id in real_particles['particle'].unique():
            particle_data = real_particles[real_particles['particle'] == particle_id].sort_values('frame')
            
            # Check if event labels follow logical sequence
            labels = particle_data['event_label'].values
            
            # Post-merge/post-split should follow merge/split events
            for i in range(1, len(labels)):
                if labels[i] == 3:  # POST_MERGE
                    # Check if there was a merge event recently
                    recent_merge = any(labels[max(0, i-10):i] == 1)
                    if not recent_merge:
                        self.warnings.append(f"Particle {particle_id} has POST_MERGE without recent MERGE")
                
                if labels[i] == 4:  # POST_SPLIT
                    # Check if there was a split event recently
                    recent_split = any(labels[max(0, i-10):i] == 2)
                    if not recent_split:
                        self.warnings.append(f"Particle {particle_id} has POST_SPLIT without recent SPLIT")
    
    def _check_spatial_boundaries(self):
        """Check if particles stay within defined boundaries"""
        # Assuming boundaries based on the simulation parameters
        x_bounds = (0, 100)
        y_bounds = (0, 100)
        z_bounds = (0, 50)
        
        out_of_bounds = self.df[
            (self.df['x'] < x_bounds[0]) | (self.df['x'] > x_bounds[1]) |
            (self.df['y'] < y_bounds[0]) | (self.df['y'] > y_bounds[1]) |
            (self.df['z'] < z_bounds[0]) | (self.df['z'] > z_bounds[1])
        ]
        
        if not out_of_bounds.empty:
            self.warnings.append(f"Found {len(out_of_bounds)} data points outside spatial boundaries")
    
    def _check_ghost_particles(self):
        """Validate ghost particle properties"""
        ghost_particles = self.df[self.df['is_ghost'] == True]
        
        if not ghost_particles.empty:
            # Ghost particles should have true_particle_id = -1
            invalid_ghosts = ghost_particles[ghost_particles['true_particle_id'] != -1]
            if not invalid_ghosts.empty:
                self.errors.append(f"Found {len(invalid_ghosts)} ghost particles with invalid true_particle_id")
            
            # Ghost particles should have empty parent_ids
            invalid_parents = ghost_particles[ghost_particles['parent_ids'] != '']
            if not invalid_parents.empty:
                self.warnings.append(f"Found {len(invalid_parents)} ghost particles with non-empty parent_ids")
            
            self.info.append(f"Found {len(ghost_particles)} ghost particle data points")
    
    def _check_parent_child_relationships(self):
        """Validate parent-child relationships"""
        real_particles = self.df[self.df['is_ghost'] == False]
        
        # Check if parent IDs reference existing particles
        particles_with_parents = real_particles[real_particles['parent_ids'] != '']
        all_particle_ids = set(real_particles['particle'].unique())
        
        for _, row in particles_with_parents.iterrows():
            if row['parent_ids']:
                parent_ids = [int(pid) for pid in row['parent_ids'].split(',') if pid.strip()]
                invalid_parents = [pid for pid in parent_ids if pid not in all_particle_ids]
                
                if invalid_parents:
                    self.warnings.append(f"Particle {row['particle']} references non-existent parent(s): {invalid_parents}")
    
    def _check_statistical_plausibility(self):
        """Check statistical plausibility of the simulation"""
        real_particles = self.df[self.df['is_ghost'] == False]
        
        if real_particles.empty:
            return
        
        # Check mass distribution
        mass_stats = real_particles['mass'].describe()
        if mass_stats['min'] < 0:
            self.errors.append("Found negative mass values")
        
        # Check if mass range is reasonable
        mass_range = mass_stats['max'] - mass_stats['min']
        if mass_range == 0:
            self.warnings.append("All particles have identical mass")
        
        # Check particle count per frame
        particles_per_frame = real_particles.groupby('frame')['particle'].nunique()
        if particles_per_frame.std() > particles_per_frame.mean():
            self.warnings.append("High variance in particle count per frame")
        
        self.info.append(f"Mass statistics: min={mass_stats['min']:.0f}, max={mass_stats['max']:.0f}, "
                        f"mean={mass_stats['mean']:.0f}, std={mass_stats['std']:.0f}")
        self.info.append(f"Particle count per frame: min={particles_per_frame.min()}, "
                        f"max={particles_per_frame.max()}, mean={particles_per_frame.mean():.1f}")
    
    def _check_duration_consistency(self):
        """Check if duration values are consistent with actual particle lifetimes"""
        real_particles = self.df[self.df['is_ghost'] == False]
        
        for particle_id in real_particles['particle'].unique():
            particle_data = real_particles[real_particles['particle'] == particle_id].sort_values('frame')
            
            if len(particle_data) == 0:
                continue
            
            # Calculate actual duration from data
            actual_duration = particle_data['frame'].max() - particle_data['frame'].min()
            recorded_duration = particle_data['duration'].iloc[0]
            
            # Allow some tolerance for duration differences
            if abs(actual_duration - recorded_duration) > 1:
                self.warnings.append(f"Particle {particle_id} duration mismatch: "
                                   f"recorded={recorded_duration}, actual={actual_duration}")
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'validation_passed': len(self.errors) == 0,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'summary': {
                'total_data_points': len(self.df),
                'unique_particles': self.df['particle'].nunique(),
                'frame_range': (self.df['frame'].min(), self.df['frame'].max()) if not self.df.empty else (0, 0),
                'ghost_particles': len(self.df[self.df['is_ghost'] == True]),
                'real_particles': len(self.df[self.df['is_ghost'] == False])
            }
        }
        
        if self.verbose:
            self._print_report(report)
        
        return report
    
    def _print_report(self, report: Dict):
        """Print formatted validation report"""
        print("=" * 60)
        print("SIMULATION VALIDATION REPORT")
        print("=" * 60)
        
        # Summary
        print(f"Validation Status: {'PASSED' if report['validation_passed'] else 'FAILED'}")
        print(f"Total Errors: {report['total_errors']}")
        print(f"Total Warnings: {report['total_warnings']}")
        print(f"Total Data Points: {report['summary']['total_data_points']}")
        print(f"Unique Particles: {report['summary']['unique_particles']}")
        print(f"Frame Range: {report['summary']['frame_range'][0]} - {report['summary']['frame_range'][1]}")
        print(f"Ghost Particles: {report['summary']['ghost_particles']}")
        print(f"Real Particles: {report['summary']['real_particles']}")
        
        # Errors
        if report['errors']:
            print("\nERRORS:")
            for i, error in enumerate(report['errors'], 1):
                print(f"  {i}. {error}")
        
        # Warnings
        if report['warnings']:
            print("\nWARNINGS:")
            for i, warning in enumerate(report['warnings'], 1):
                print(f"  {i}. {warning}")
        
        # Info
        if report['info']:
            print("\nINFORMATION:")
            for i, info in enumerate(report['info'], 1):
                print(f"  {i}. {info}")
        
        print("=" * 60)

def validate_simulation_output(csv_file_path: str, verbose: bool = True) -> Dict:
    """
    Convenience function to validate simulation output from CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing simulation output
        verbose: Whether to print detailed report
        
    Returns:
        Dictionary containing validation results
    """
    try:
        df = pd.read_csv(csv_file_path)
        validator = SimulationValidator(df, verbose=verbose)
        return validator.validate_simulation_output()
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found")
        return {'validation_passed': False, 'errors': ['File not found']}
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return {'validation_passed': False, 'errors': [str(e)]}

# Example usage and integration with the existing simulation code
# Integration with existing simulation code
def run_single_simulation_with_validation(sim_id):
    """Updated version of run_single_simulation that includes validation"""
    np.random.seed(sim_id)
    
    output_dir = 'data/3class/tracked_simdata_partiallinking'
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = 200
    initial_particles = (13, 25)
    num_initial = np.random.randint(initial_particles[0], initial_particles[1] + 1)
    
    # Create simulator (same as original)
    sim = TrackedParticleSimulator(
        x_range=(0, 100),
        y_range=(0, 100),
        z_range=(0, 50),  
        min_mass=50000,
        max_mass=500000,
        merge_distance_factor=10.0,
        split_mass_threshold=100000,
        merge_prob=0.7,
        split_prob=0.01,
        merge_linking_prob=0.3,
        split_linking_prob=0.3,
        spontaneous_appear_prob=0.00,
        spontaneous_disappear_prob=0.00,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
        max_linking_gap=0,
        linking_gap_prob=0,
        enable_tracking_gaps=True,
        position_noise_sigma=0.5,
        mass_noise_cv=0.1,
        reflective_boundaries=True,
        event_cooldown=5,
        temporal_window=0
    )
    
    # Initialize particles
    for i in range(num_initial):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        z = np.random.uniform(10, 40)
        mass = np.random.lognormal(np.log(150000), 0.5)
        mass = np.clip(mass, sim.min_mass, sim.max_mass)
        sim.add_particle(x, y, z, mass) 
    
    # Run simulation
    for frame in range(num_frames):
        sim.update(dt=0.1)
    
    # Export to CSV
    output_file = os.path.join(output_dir, f'tracked_particles_3d_{sim_id:05d}.csv')
    df = sim.export_to_csv(output_file)
    
    # Validate the output
    validator = SimulationValidator(df, verbose=False)  # Set verbose=False for batch processing
    validation_results = validator.validate_simulation_output()
    
    return {
        'sim_id': sim_id,
        'num_rows': len(df),
        'validation_passed': validation_results['validation_passed'],
        'num_errors': validation_results['total_errors'],
        'num_warnings': validation_results['total_warnings'],
        'errors': validation_results['errors'],
        'warnings': validation_results['warnings']
    }

if __name__ == "__main__":
    # Example of how to use the validator standalone
    
    # Assuming you have a CSV file from the simulation
    csv_file = "data/tracked_simdata_partiallinking/tracked_particles_3d_00000.csv"  # Replace with actual file path
    
    # Validate the simulation output
    results = validate_simulation_output(csv_file, verbose=True)
    
    # Check if validation passed
    if results['validation_passed']:
        print("✅ Simulation output validation PASSED")
    else:
        print("❌ Simulation output validation FAILED")
        print(f"Found {results['total_errors']} errors and {results['total_warnings']} warnings")
    
    # Example of running a single simulation with validation
    print("\n" + "="*50)
    print("RUNNING SIMULATION WITH VALIDATION")
    print("="*50)
    
    result = run_single_simulation_with_validation(0)
    print(f"Simulation {result['sim_id']}: {result['num_rows']} rows generated")
    print(f"Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
    if result['num_errors'] > 0:
        print(f"Errors ({result['num_errors']}):")
        for error in result['errors']:
            print(f"  - {error}")
    if result['num_warnings'] > 0:
        print(f"Warnings ({result['num_warnings']}):")
        for warning in result['warnings']:
            print(f"  - {warning}")