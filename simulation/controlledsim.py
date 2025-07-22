import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from enum import IntEnum
from tqdm import tqdm
import os
from typing import List, Tuple, Dict, Optional, Set
from multiprocessing import Pool, cpu_count
import time
import warnings

class EventLabel(IntEnum):
    NORMAL = 0
    MERGE = 1
    SPLIT = 2
    POST_MERGE = 3
    POST_SPLIT = 4

class ControlledParticleSimulator:
    """
    Controlled particle simulator for generating balanced training data.
    Maintains realistic physics while ensuring controlled event generation.
    """
    def __init__(self, 
                 x_range=(0, 200), y_range=(0, 200), z_range=(0, 200),
                 min_mass=50000, max_mass=500000,
                 temperature=300, viscosity=0.1, pixel_size=100,
                 merge_distance_factor=2.0, split_mass_threshold=100000,
                 position_noise_sigma=0.5, mass_noise_cv=0.1,
                 reflective_boundaries=True, event_cooldown=5,
                 enable_warnings=False, detection_prob=0.95,
                 # Controlled event parameters
                 target_event_counts=None,
                 event_spacing=10,
                 force_event_success=True,
                 maintain_physics=True):
        
        # Base simulation parameters (same as original)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.min_mass = min_mass
        self.max_mass = max_mass
        
        # Physics constants
        self.kB = 1.38e-23
        self.T = temperature
        self.eta = viscosity
        self.pixel_size = pixel_size
        
        # Event parameters
        self.merge_distance_factor = merge_distance_factor
        self.split_mass_threshold = split_mass_threshold
        self.position_noise_sigma = position_noise_sigma
        self.mass_noise_cv = mass_noise_cv
        self.reflective_boundaries = reflective_boundaries
        self.event_cooldown = event_cooldown
        self.enable_warnings = enable_warnings
        self.detection_prob = detection_prob
        
        # Controlled event parameters
        if target_event_counts is None:
            # Default: balanced dataset
            target_event_counts = {
                EventLabel.NORMAL: 1000,
                EventLabel.MERGE: 200,
                EventLabel.SPLIT: 200,
                EventLabel.POST_MERGE: 200,
                EventLabel.POST_SPLIT: 200
            }
        self.target_event_counts = target_event_counts
        self.event_spacing = event_spacing  # Minimum frames between forced events
        self.force_event_success = force_event_success
        self.maintain_physics = maintain_physics
        
        # State tracking
        self.particles = []
        self.next_id = 0
        self.current_frame = 0
        self.event_counts = {label: 0 for label in EventLabel}
        self.scheduled_events = []  # Queue of events to execute
        self.last_forced_event_frame = -float('inf')
        
        # Physics state
        self.all_merges = []
        self.all_splits = []
        self.frame_warnings = []
        self.pending_post_events = []

    def log_warning(self, message):
        if self.enable_warnings:
            warning_msg = f"FRAME {self.current_frame}: {message}"
            print(f"WARNING: {warning_msg}")
            self.frame_warnings.append(warning_msg)

    def add_particle(self, x, y, z, mass, parent_ids=None, birth_frame=None):
        """Add particle with same physics as original simulator"""
        if birth_frame is None:
            birth_frame = self.current_frame
        if parent_ids is None:
            parent_ids = []
            
        # Physics calculations
        volume = mass / 100000
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 100
        D = self.kB * self.T / (6 * np.pi * self.eta * radius * 1e-9)
        is_active = birth_frame <= self.current_frame
        
        particle = {
            'id': self.next_id,
            'parent_ids': parent_ids.copy(),
            'x': x, 'y': y, 'z': z, 'mass': mass, 'radius': radius,
            'D': D * 1e18 / (self.pixel_size**2),
            'active': is_active,
            'birth_frame': birth_frame, 'death_frame': None,
            'trajectory': [], 'mass_history': [], 'label_history': [],
            'last_event_frame': -float('inf'),
            'recorded_this_frame': False,
            'detectable': True, 'detection_prob': self.detection_prob,
            'forced_event_pending': None  # For controlled events
        }
        
        self.particles.append(particle)
        self.next_id += 1
        return particle['id']

    def record_particle_state(self, particle, label):
        """Record particle state and update event counts"""
        if particle.get('recorded_this_frame', False):
            return
            
        if not particle['active']:
            return
            
        particle['trajectory'].append((particle['x'], particle['y'], particle['z'], self.current_frame))
        particle['mass_history'].append(particle['mass'])
        particle['label_history'].append(label)
        particle['recorded_this_frame'] = True
        
        # Update event counts
        self.event_counts[label] += 1

    def update_particle_position(self, particle, dt=0.1):
        """Same diffusion physics as original"""
        if not particle['active']:
            return False
            
        dx = np.random.normal(0, np.sqrt(2 * particle['D'] * dt))
        dy = np.random.normal(0, np.sqrt(2 * particle['D'] * dt))
        dz = np.random.normal(0, np.sqrt(2 * particle['D'] * dt))
        
        new_x = particle['x'] + dx
        new_y = particle['y'] + dy
        new_z = particle['z'] + dz
        
        if self.reflective_boundaries:
            new_x = np.clip(new_x, self.x_range[0], self.x_range[1])
            new_y = np.clip(new_y, self.y_range[0], self.y_range[1])
            new_z = np.clip(new_z, self.z_range[0], self.z_range[1])
        
        particle['x'] = new_x
        particle['y'] = new_y
        particle['z'] = new_z
        return False

    def schedule_event(self, event_type, target_frame, particle_ids=None):
        """Schedule a controlled event for execution"""
        self.scheduled_events.append({
            'type': event_type,
            'frame': target_frame,
            'particle_ids': particle_ids
        })

    def create_merge_scenario(self, target_frame=None):
        """Create particles positioned for a controlled merge"""
        if target_frame is None:
            target_frame = self.current_frame + 5
            
        # Create two particles that will merge
        center_x = np.random.uniform(self.x_range[0] + 20, self.x_range[1] - 20)
        center_y = np.random.uniform(self.y_range[0] + 20, self.y_range[1] - 20)
        center_z = np.random.uniform(self.z_range[0] + 10, self.z_range[1] - 10)
        
        # Calculate separation distance for controlled collision
        mass1 = np.random.uniform(self.min_mass, self.max_mass * 0.4)
        mass2 = np.random.uniform(self.min_mass, self.max_mass * 0.4)
        
        volume1 = mass1 / 100000
        radius1 = np.cbrt(3 * volume1 / (4 * np.pi)) * 100
        volume2 = mass2 / 100000
        radius2 = np.cbrt(3 * volume2 / (4 * np.pi)) * 100
        
        # Start particles at controlled distance
        merge_distance = self.merge_distance_factor * (radius1 + radius2) / (2 * self.pixel_size)
        initial_separation = merge_distance * 2.5  # Start slightly apart
        
        angle = np.random.uniform(0, 2 * np.pi)
        dx = initial_separation * np.cos(angle) / 2
        dy = initial_separation * np.sin(angle) / 2
        
        p1_id = self.add_particle(center_x - dx, center_y - dy, center_z, mass1)
        p2_id = self.add_particle(center_x + dx, center_y + dy, center_z, mass2)
        
        # Schedule merge event
        self.schedule_event('FORCE_MERGE', target_frame, [p1_id, p2_id])
        
        return p1_id, p2_id

    def create_split_scenario(self, target_frame=None):
        """Create particle positioned for a controlled split"""
        if target_frame is None:
            target_frame = self.current_frame + 3
            
        # Create particle with mass suitable for splitting
        x = np.random.uniform(self.x_range[0] + 20, self.x_range[1] - 20)
        y = np.random.uniform(self.y_range[0] + 20, self.y_range[1] - 20)
        z = np.random.uniform(self.z_range[0] + 10, self.z_range[1] - 10)
        
        # Ensure mass is above split threshold
        mass = np.random.uniform(self.split_mass_threshold * 1.2, self.max_mass * 0.8)
        
        p_id = self.add_particle(x, y, z, mass)
        
        # Schedule split event
        self.schedule_event('FORCE_SPLIT', target_frame, [p_id])
        
        return p_id

    def execute_forced_merge(self, particle_ids):
        """Execute a controlled merge event"""
        p1 = next((p for p in self.particles if p['id'] == particle_ids[0]), None)
        p2 = next((p for p in self.particles if p['id'] == particle_ids[1]), None)
        
        if not p1 or not p2 or not p1['active'] or not p2['active']:
            return False
            
        # Position particles for merge if needed
        if self.force_event_success:
            distance = np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)
            merge_threshold = self.merge_distance_factor * (p1['radius'] + p2['radius']) / (2 * self.pixel_size)
            
            if distance > merge_threshold:
                # Move particles closer
                midpoint_x = (p1['x'] + p2['x']) / 2
                midpoint_y = (p1['y'] + p2['y']) / 2
                midpoint_z = (p1['z'] + p2['z']) / 2
                
                offset = merge_threshold * 0.8 / 2
                p1['x'] = midpoint_x - offset
                p1['y'] = midpoint_y
                p2['x'] = midpoint_x + offset
                p2['y'] = midpoint_y
        
        return self.process_controlled_merge(p1, p2)

    def execute_forced_split(self, particle_ids):
        """Execute a controlled split event"""
        particle = next((p for p in self.particles if p['id'] == particle_ids[0]), None)
        
        if not particle or not particle['active']:
            return False
            
        # Ensure mass is sufficient for split
        if self.force_event_success and particle['mass'] < self.split_mass_threshold:
            particle['mass'] = self.split_mass_threshold * 1.1
            
        return self.process_controlled_split(particle)

    def process_controlled_merge(self, p1, p2):
        """Process merge with controlled parameters"""
        total_mass = p1['mass'] + p2['mass']
        new_x = (p1['x'] * p1['mass'] + p2['x'] * p2['mass']) / total_mass
        new_y = (p1['y'] * p1['mass'] + p2['y'] * p2['mass']) / total_mass
        new_z = (p1['z'] * p1['mass'] + p2['z'] * p2['mass']) / total_mass
        
        # Record merge events
        self.record_particle_state(p1, EventLabel.MERGE)
        self.record_particle_state(p2, EventLabel.MERGE)
        
        # Choose merge type (linking vs normal)
        if np.random.random() < 0.5:  # 50% chance of each type
            # Linking merge
            continuing_particle = p1 if np.random.random() < 0.5 else p2
            disappearing_particle = p2 if continuing_particle == p1 else p1
            
            disappearing_particle['active'] = False
            disappearing_particle['death_frame'] = self.current_frame
            
            # Update continuing particle
            continuing_particle['x'] = new_x
            continuing_particle['y'] = new_y
            continuing_particle['z'] = new_z
            continuing_particle['mass'] = total_mass
            continuing_particle['last_event_frame'] = self.current_frame
            
            # Schedule post-merge event
            self.pending_post_events.append({
                'type': 'POST_MERGE',
                'particle_ids': [continuing_particle['id']]
            })
            
        else:
            # Normal merge
            p1['active'] = False
            p1['death_frame'] = self.current_frame
            p2['active'] = False
            p2['death_frame'] = self.current_frame
            
            # Create new particle
            new_id = self.add_particle(new_x, new_y, new_z, total_mass, 
                                     parent_ids=[p1['id'], p2['id']], 
                                     birth_frame=self.current_frame + 1)
            
            # Schedule post-merge event
            self.pending_post_events.append({
                'type': 'POST_MERGE',
                'particle_ids': [new_id]
            })
        
        self.all_merges.append({
            'parent_ids': [p1['id'], p2['id']],
            'frame': self.current_frame
        })
        
        return True

    def process_controlled_split(self, particle):
        """Process split with controlled parameters"""
        # Record split event
        self.record_particle_state(particle, EventLabel.SPLIT)
        
        # Calculate split masses
        ratio = np.random.uniform(0.4, 0.6)
        mass1 = particle['mass'] * ratio
        mass2 = particle['mass'] * (1 - ratio)
        
        # Calculate split positions
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        sep_dist = np.random.uniform(3, 8)
        
        dx = sep_dist * np.sin(theta) * np.cos(phi)
        dy = sep_dist * np.sin(theta) * np.sin(phi)
        dz = sep_dist * np.cos(theta)
        
        # Choose split type (linking vs normal)
        if np.random.random() < 0.5:  # 50% chance of each type
            # Linking split
            continuing_mass, new_mass = (mass1, mass2) if mass1 >= mass2 else (mass2, mass1)
            
            # Update original particle
            particle['x'] += -dx/2 if mass1 >= mass2 else dx/2
            particle['y'] += -dy/2 if mass1 >= mass2 else dy/2
            particle['z'] += -dz/2 if mass1 >= mass2 else dz/2
            particle['mass'] = continuing_mass
            particle['last_event_frame'] = self.current_frame
            
            # Create new particle
            new_x = particle['x'] + (dx if mass1 >= mass2 else -dx)
            new_y = particle['y'] + (dy if mass1 >= mass2 else -dy)
            new_z = particle['z'] + (dz if mass1 >= mass2 else -dz)
            
            new_id = self.add_particle(new_x, new_y, new_z, new_mass,
                                     parent_ids=particle['parent_ids'].copy(),
                                     birth_frame=self.current_frame + 1)
            
            # Schedule post-split events
            self.pending_post_events.append({
                'type': 'POST_SPLIT',
                'particle_ids': [particle['id'], new_id]
            })
            
        else:
            # Normal split
            particle['active'] = False
            particle['death_frame'] = self.current_frame
            
            # Create two new particles
            child1_id = self.add_particle(
                particle['x'] - dx/2, particle['y'] - dy/2, particle['z'] - dz/2,
                mass1, parent_ids=[particle['id']], birth_frame=self.current_frame + 1
            )
            child2_id = self.add_particle(
                particle['x'] + dx/2, particle['y'] + dy/2, particle['z'] + dz/2,
                mass2, parent_ids=[particle['id']], birth_frame=self.current_frame + 1
            )
            
            # Schedule post-split events
            self.pending_post_events.append({
                'type': 'POST_SPLIT',
                'particle_ids': [child1_id, child2_id]
            })
        
        self.all_splits.append({
            'parent_id': particle['id'],
            'frame': self.current_frame
        })
        
        return True

    def process_pending_post_events(self):
        """Process pending post-events (POST_MERGE, POST_SPLIT)"""
        # Activate particles born this frame
        for particle in self.particles:
            if not particle['active'] and particle['birth_frame'] == self.current_frame:
                particle['active'] = True
        
        # Process pending post-events
        for event in self.pending_post_events:
            for particle_id in event['particle_ids']:
                particle = next((p for p in self.particles if p['id'] == particle_id), None)
                if particle and particle['active']:
                    if event['type'] == 'POST_MERGE':
                        self.record_particle_state(particle, EventLabel.POST_MERGE)
                    elif event['type'] == 'POST_SPLIT':
                        self.record_particle_state(particle, EventLabel.POST_SPLIT)
        
        self.pending_post_events = []

    def should_create_event(self):
        """Determine if we should create a new event based on target counts"""
        # Check if we need more events
        for event_type, target_count in self.target_event_counts.items():
            if event_type == EventLabel.NORMAL:
                continue  # Normal events happen naturally
                
            current_count = self.event_counts[event_type]
            if current_count < target_count:
                # Check if enough time has passed since last forced event
                if self.current_frame - self.last_forced_event_frame >= self.event_spacing:
                    return event_type
        
        return None

    def update(self, dt=0.1):
        """Main update loop with controlled event generation"""
        self.current_frame += 1
        
        # Reset recording flags
        for particle in self.particles:
            particle['recorded_this_frame'] = False
        
        # Process pending post-events first
        self.process_pending_post_events()
        
        # Check if we need to create new events
        needed_event = self.should_create_event()
        if needed_event is not None:
            active_particles = [p for p in self.particles if p['active']]
            
            if needed_event in [EventLabel.MERGE, EventLabel.POST_MERGE]:
                if len(active_particles) >= 2:
                    self.create_merge_scenario()
                    self.last_forced_event_frame = self.current_frame
                elif len(active_particles) < 2:
                    # Need more particles for merge
                    self.create_additional_particles(2)
                    
            elif needed_event in [EventLabel.SPLIT, EventLabel.POST_SPLIT]:
                suitable_particles = [p for p in active_particles 
                                    if p['mass'] >= self.split_mass_threshold]
                if suitable_particles:
                    self.create_split_scenario()
                    self.last_forced_event_frame = self.current_frame
                else:
                    # Create particle suitable for splitting
                    self.create_splittable_particle()
        
        # Execute scheduled events
        events_to_execute = [e for e in self.scheduled_events if e['frame'] == self.current_frame]
        for event in events_to_execute:
            if event['type'] == 'FORCE_MERGE':
                self.execute_forced_merge(event['particle_ids'])
            elif event['type'] == 'FORCE_SPLIT':
                self.execute_forced_split(event['particle_ids'])
        
        # Remove executed events
        self.scheduled_events = [e for e in self.scheduled_events if e['frame'] != self.current_frame]
        
        # Update remaining particles normally
        unrecorded_particles = [p for p in self.particles 
                              if p['active'] and not p.get('recorded_this_frame', False)]
        
        for particle in unrecorded_particles:
            # Update position with diffusion
            self.update_particle_position(particle, dt)
            
            # Record as normal event
            self.record_particle_state(particle, EventLabel.NORMAL)

    def create_additional_particles(self, count):
        """Create additional particles for event generation"""
        for _ in range(count):
            x = np.random.uniform(self.x_range[0] + 10, self.x_range[1] - 10)
            y = np.random.uniform(self.y_range[0] + 10, self.y_range[1] - 10)
            z = np.random.uniform(self.z_range[0] + 5, self.z_range[1] - 5)
            mass = np.random.uniform(self.min_mass, self.max_mass * 0.5)
            self.add_particle(x, y, z, mass)

    def create_splittable_particle(self):
        """Create a particle suitable for splitting"""
        x = np.random.uniform(self.x_range[0] + 20, self.x_range[1] - 20)
        y = np.random.uniform(self.y_range[0] + 20, self.y_range[1] - 20)
        z = np.random.uniform(self.z_range[0] + 10, self.z_range[1] - 10)
        mass = np.random.uniform(self.split_mass_threshold * 1.5, self.max_mass * 0.8)
        return self.add_particle(x, y, z, mass)

    def is_simulation_complete(self):
        """Check if target event counts have been reached"""
        for event_type, target_count in self.target_event_counts.items():
            if self.event_counts[event_type] < target_count:
                return False
        return True

    def export_to_csv(self, filename):
        """Export simulation data to CSV (same format as original)"""
        data = []
        
        for particle in self.particles:
            if not particle['trajectory']:
                continue
                
            death_frame = particle['death_frame'] if particle['death_frame'] is not None else self.current_frame
            
            for i, trajectory_entry in enumerate(particle['trajectory']):
                x, y, z, frame = trajectory_entry
                
                # Add noise like original simulator
                noisy_x = x + np.random.normal(0, self.position_noise_sigma)
                noisy_y = y + np.random.normal(0, self.position_noise_sigma)
                noisy_z = z + np.random.normal(0, self.position_noise_sigma)
                
                if i >= len(particle['mass_history']) or i >= len(particle['label_history']):
                    continue
                    
                noisy_mass = particle['mass_history'][i] * np.random.normal(1, self.mass_noise_cv)
                event_label = particle['label_history'][i]
                
                total_duration = death_frame - particle['birth_frame'] + 1
                detected_duration = len(particle['trajectory'])

                data.append({
                    'frame': float(frame),
                    'x': noisy_x, 'y': noisy_y, 'z': noisy_z,
                    'mass': noisy_mass,
                    'particle': float(particle['id']),
                    'duration': float(total_duration),
                    'detected_duration': float(detected_duration),
                    'event_label': int(event_label),
                    'true_particle_id': float(particle['id']),
                    'parent_ids': ','.join(map(str, particle['parent_ids'])) if particle['parent_ids'] else '',
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(['frame', 'particle'])
        
        df.to_csv(filename, index=False)
        return df

    def print_summary(self):
        """Print simulation summary"""
        print(f"\n=== CONTROLLED SIMULATION SUMMARY ===")
        print(f"Total frames: {self.current_frame}")
        print(f"Total particles created: {self.next_id}")
        
        print(f"\n=== EVENT COUNTS ===")
        for event_type in EventLabel:
            current = self.event_counts[event_type]
            target = self.target_event_counts.get(event_type, 0)
            print(f"{event_type.name}: {current}/{target} ({current/max(target,1)*100:.1f}%)")
        
        print(f"\n=== PHYSICS EVENTS ===")
        print(f"Total merges: {len(self.all_merges)}")
        print(f"Total splits: {len(self.all_splits)}")
        print("====================================\n")


def run_controlled_simulation(sim_id, target_counts=None):
    """Run a single controlled simulation"""
    np.random.seed(sim_id)
    
    if target_counts is None:
        # Balanced dataset - equal representation
        target_counts = {
            EventLabel.NORMAL: 800,
            EventLabel.MERGE: 200,
            EventLabel.SPLIT: 200,
            EventLabel.POST_MERGE: 200,
            EventLabel.POST_SPLIT: 200
        }
    
    sim = ControlledParticleSimulator(
        x_range=(0, 100), y_range=(0, 100), z_range=(0, 50),
        min_mass=50000, max_mass=500000,
        merge_distance_factor=3.0, split_mass_threshold=100000,
        position_noise_sigma=0.5, mass_noise_cv=0.1,
        reflective_boundaries=True, event_cooldown=3,
        enable_warnings=False, detection_prob=0.98,
        target_event_counts=target_counts,
        event_spacing=8,
        force_event_success=True
    )
    
    # Initialize with fewer particles (will create more as needed)
    initial_particles = 5
    for i in range(initial_particles):
        x = np.random.uniform(20, 80)
        y = np.random.uniform(20, 80)
        z = np.random.uniform(10, 40)
        mass = np.random.uniform(sim.min_mass, sim.max_mass * 0.6)
        particle_id = sim.add_particle(x, y, z, mass)
        
        particle = next(p for p in sim.particles if p['id'] == particle_id)
        sim.record_particle_state(particle, EventLabel.NORMAL)
    
    # Run simulation until targets are met or max frames reached
    max_frames = 2000
    for frame in range(max_frames):
        sim.update(dt=0.1)
        
        if sim.is_simulation_complete():
            if sim.enable_warnings:
                print(f"Simulation {sim_id} completed at frame {sim.current_frame}")
            break
    
    return sim


def generate_balanced_dataset(num_simulations=100, output_dir='data/controlled_synthetic'):
    """Generate balanced synthetic dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_simulations} controlled simulations...")
    
    all_data = []
    
    for sim_id in tqdm(range(num_simulations), desc="Controlled Simulations"):
        sim = run_controlled_simulation(sim_id)
        
        # Export individual file
        output_file = os.path.join(output_dir, f'controlled_particles_{sim_id:05d}.csv')
        df = sim.export_to_csv(output_file)
        all_data.append(df)
        
        if sim_id == 0:
            sim.print_summary()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_file = os.path.join(output_dir, 'combined_controlled_dataset.csv')
    combined_df.to_csv(combined_file, index=False)
    
    # Print final statistics
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total samples: {len(combined_df)}")
    print(f"Event distribution:")
    for event_type in EventLabel:
        count = (combined_df['event_label'] == event_type.value).sum()
        percent = count / len(combined_df) * 100
        print(f"  {event_type.name}: {count} ({percent:.1f}%)")
    
    return combined_df


if __name__ == "__main__":
    # Generate balanced synthetic dataset
    dataset = generate_balanced_dataset(num_simulations=50, 
                                       output_dir='../data/controlled_synthetic_balanced')
    
    print("Controlled synthetic dataset generation complete!")