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
import random
import math
class EventLabel(IntEnum):
    NORMAL = 0
    MERGE = 1
    SPLIT = 2
    POST_MERGE = 3
    POST_SPLIT = 4


class TrackedParticleSimulator:
    def __init__(self, 
                 x_range=(0, 200),
                 y_range=(0, 200),
                 z_range=(0, 200),
                 min_mass=50000,
                 max_mass=500000,
                 temperature=300,
                 viscosity=0.1,
                 pixel_size=100,
                 merge_distance_factor=2.0,
                 split_mass_threshold=100000,
                 merge_prob=0.7,
                 split_prob=0.005,
                 merge_linking_prob=0.8,
                 split_linking_prob=0.8,
                 spontaneous_appear_prob=0.002,
                 spontaneous_disappear_prob=0.001,
                 position_noise_sigma=0.5,
                 mass_noise_cv=0.1,
                 reflective_boundaries=True,
                 event_cooldown=5):
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.min_mass = min_mass
        self.max_mass = max_mass
        
        self.kB = 1.38e-23
        self.T = temperature
        self.eta = viscosity
        self.pixel_size = pixel_size
        
        self.merge_distance_factor = merge_distance_factor
        self.split_mass_threshold = split_mass_threshold
        self.merge_prob = merge_prob
        self.split_prob = split_prob
        
        self.merge_linking_prob = merge_linking_prob
        self.split_linking_prob = split_linking_prob
        
        self.spontaneous_appear_prob = spontaneous_appear_prob
        self.spontaneous_disappear_prob = spontaneous_disappear_prob
        
        self.position_noise_sigma = position_noise_sigma
        self.mass_noise_cv = mass_noise_cv
        
        self.reflective_boundaries = reflective_boundaries
        
        self.event_cooldown = event_cooldown
        
        self.particles = []
        self.next_id = 0
        self.current_frame = 0
        
        self.current_frame_events = {'merges': [], 'splits': []}
        self.recent_event_particles = {}

    def record_particle_state(self, particle, label):
        """Record the current state of a particle"""
        particle['trajectory'].append((particle['x'], particle['y'], particle['z']))
        particle['mass_history'].append(particle['mass'])
        particle['label_history'].append(label)
        particle['last_recorded_frame'] = self.current_frame

    def add_particle(self, x, y, z, mass, parent_ids=None, birth_frame=None, inherit_cooldown=False):
        if birth_frame is None:
            birth_frame = self.current_frame
        
        if parent_ids is None:
            parent_ids = []
            
        volume = mass / 100000
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 100
        
        D = self.kB * self.T / (6 * np.pi * self.eta * radius * 1e-9)
        
        # If inherit_cooldown is True and particle has parents, set last_event_frame to current frame
        if inherit_cooldown and parent_ids:
            last_event_frame = self.current_frame
        else:
            last_event_frame = -float('inf')
        
        particle = {
            'id': self.next_id,
            'parent_ids': parent_ids.copy(),
            'x': x,
            'y': y,
            'z': z,
            'mass': mass,
            'radius': radius,
            'D': D * 1e18 / (self.pixel_size**2),
            'active': True,
            'birth_frame': birth_frame,
            'death_frame': None,
            'trajectory': [],
            'mass_history': [],
            'label_history': [],
            'last_event_frame': last_event_frame,
            'last_updated_frame': self.current_frame,
            'last_recorded_frame': -1
        }
        
        self.particles.append(particle)
        self.next_id += 1
        
        return particle['id']
    
    def update_existing_particle(self, particle, x, y, z, mass, parent_ids):
        """Update an existing particle's properties"""
        volume = mass / 100000
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 100
        D = self.kB * self.T / (6 * np.pi * self.eta * radius * 1e-9)
        
        particle['x'] = x
        particle['y'] = y
        particle['z'] = z
        particle['mass'] = mass
        particle['radius'] = radius
        particle['D'] = D * 1e18 / (self.pixel_size**2)
        particle['parent_ids'] = parent_ids.copy()
        particle['last_updated_frame'] = self.current_frame

    def can_have_event(self, particle):
        if self.current_frame - particle.get('last_event_frame', -float('inf')) <= self.event_cooldown:
            print(f"Suppressed {particle['id']} due to last_event_frame")
            return False
        if particle['id'] in self.recent_event_particles:
            if self.current_frame - self.recent_event_particles[particle['id']] <= self.event_cooldown:
                print(f"Suppressed {particle['id']} due to recent_event_particles")
                return False
        if particle['parent_ids'] and self.current_frame - particle['birth_frame'] <= self.event_cooldown:
            print(f"Suppressed {particle['id']} due to birth_frame")
            return False
        return True

    
    def update_particle_position(self, particle, dt=0.1):
        """Update particle position with Brownian motion"""
        if not particle['active']:
            return False
            
        dx = np.random.normal(0, np.sqrt(2 * particle['D'] * dt))
        dy = np.random.normal(0, np.sqrt(2 * particle['D'] * dt))
        dz = np.random.normal(0, np.sqrt(2 * particle['D'] * dt))
        
        new_x = particle['x'] + dx
        new_y = particle['y'] + dy
        new_z = particle['z'] + dz
        
        if self.reflective_boundaries:
            if new_x < self.x_range[0]:
                new_x = 2 * self.x_range[0] - new_x
            elif new_x > self.x_range[1]:
                new_x = 2 * self.x_range[1] - new_x
                
            if new_y < self.y_range[0]:
                new_y = 2 * self.y_range[0] - new_y
            elif new_y > self.y_range[1]:
                new_y = 2 * self.y_range[1] - new_y
            
            if new_z < self.z_range[0]:
                new_z = 2 * self.z_range[0] - new_z
            elif new_z > self.z_range[1]:
                new_z = 2 * self.z_range[1] - new_z
            
            new_x = np.clip(new_x, self.x_range[0], self.x_range[1])
            new_y = np.clip(new_y, self.y_range[0], self.y_range[1])
            new_z = np.clip(new_z, self.z_range[0], self.z_range[1])
        else:
            if (new_x < self.x_range[0] or new_x > self.x_range[1] or 
                new_y < self.y_range[0] or new_y > self.y_range[1] or
                new_z < self.z_range[0] or new_z > self.z_range[1]):
                return True
        
        particle['x'] = new_x
        particle['y'] = new_y
        particle['z'] = new_z
        particle['last_updated_frame'] = self.current_frame
        return False
    



    def _distance(self,p1, p2):
        """
        Calculate the Euclidean distance between two 3D points.
        
        Args:
            p1: First point as a tuple/list of coordinates (x1, y1, z1)
            p2: Second point as a tuple/list of coordinates (x2, y2, z2)
        
        Returns:
            The Euclidean distance between p1 and p2
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def process_merge_phase(self, active_particles):
        """
        Phase 1: Merge processing.
        Identifies and processes merges between particles.
        90% of the time, the merged particle inherits one parent's ID (50/50 chance which one).
        """
        merged_info = []
        merge_distance_threshold = 10.0
        merged_ids = set()

        for i, p1 in enumerate(active_particles):
            if p1['id'] in merged_ids or p1['id'] in self.recent_event_particles:
                continue

            for j in range(i + 1, len(active_particles)):
                p2 = active_particles[j]
                if p2['id'] in merged_ids or p2['id'] in self.recent_event_particles:
                    continue

                # Check distance
                dist = self._distance(p1['position'], p2['position'])
                if dist > merge_distance_threshold:
                    continue

                # Compute merged attributes
                new_position = self._weighted_average(p1, p2)
                new_velocity = self._weighted_average(p1, p2, key='velocity')
                new_mass = p1['mass'] + p2['mass']

                # Determine ID inheritance logic (90% of time inherit a parent ID, 50/50 chance)
                inherit_id = random.random() < 0.9
                if inherit_id:
                    merged_id = random.choice([p1['id'], p2['id']])
                else:
                    merged_id = self.next_id
                    self.next_id += 1

                merged_particle = {
                    'id': merged_id,
                    'position': new_position,
                    'mass': new_mass,
                    'velocity': new_velocity,
                    'active': True,
                    'parents': [p1['id'], p2['id']],
                    'frame_created': self.current_frame,
                    'true_id': min(p1.get('true_id', p1['id']), p2.get('true_id', p2['id'])),
                    'merge_depth': max(p1.get('merge_depth', 0), p2.get('merge_depth', 0)) + 1,
                }

                self.particles.append(merged_particle)

                self.current_frame_events['merges'].append((p1['id'], p2['id'], merged_particle['id']))
                p1['active'] = False
                p2['active'] = False
                merged_ids.update([p1['id'], p2['id']])

                self.recent_event_particles[p1['id']] = self.current_frame
                self.recent_event_particles[p2['id']] = self.current_frame

                merged_info.append(merged_particle)
                break  # Only merge once per particle

        return merged_info



    def process_split_phase(self, active_particles, merged_info):
        """
        Phase 2: Split processing.
        Some recently merged particles may split into two, with one child usually inheriting the parent's ID.
        """
        split_info = []
        split_probability = 0.3  # 30% chance to split

        for particle in merged_info:
            if random.random() > split_probability:
                continue  # Skip this one

            # Split mass with uneven ratio
            mass_total = particle['mass']
            mass_ratio = random.uniform(0.3, 0.7)
            mass1 = mass_total * mass_ratio
            mass2 = mass_total - mass1

            pos = particle['position']
            vel = particle['velocity']
            offset = random.uniform(-2, 2)

            inherit_parent_id = random.random() < 0.9  # 90% chance to inherit parent ID

            if inherit_parent_id:
                # Randomly choose which child gets the parent's ID
                if random.random() < 0.5:
                    id1 = particle['id']  # Inherit parent ID
                    id2 = self.next_id
                    self.next_id += 1
                else:
                    id1 = self.next_id
                    self.next_id += 1
                    id2 = particle['id']
            else:
                # Both children get new IDs
                id1 = self.next_id
                self.next_id += 1
                id2 = self.next_id
                self.next_id += 1

            # Create child particles
            child1 = {
                'id': id1,
                'true_id': particle['true_id'],
                'position': (pos[0] + offset, pos[1] + offset),
                'velocity': (vel[0] + offset * 0.1, vel[1] + offset * 0.1),
                'mass': mass1,
                'active': True,
                'parents': [particle['id']],
                'frame_created': self.current_frame,
                'split_depth': particle.get('split_depth', 0) + 1,
                'merge_depth': particle.get('merge_depth', 0),
            }

            child2 = {
                'id': id2,
                'true_id': particle['true_id'],
                'position': (pos[0] - offset, pos[1] - offset),
                'velocity': (vel[0] - offset * 0.1, vel[1] - offset * 0.1),
                'mass': mass2,
                'active': True,
                'parents': [particle['id']],
                'frame_created': self.current_frame,
                'split_depth': particle.get('split_depth', 0) + 1,
                'merge_depth': particle.get('merge_depth', 0),
            }

            # Add children to particle list and deactivate parent
            self.particles.append(child1)
            self.particles.append(child2)
            particle['active'] = False
            self.recent_event_particles[particle['id']] = self.current_frame

            self.current_frame_events['splits'].append((particle['id'], child1['id'], child2['id']))
            split_info.append((child1, child2))

        return split_info

    def process_normal_update_phase(self, active_particles, merged_info, split_info, dt=0.1):
        """Phase 3: Process normal updates for particles not deactivated in merge/split events"""
        # Only exclude particles that were DEACTIVATED, not those that continued
        deactivated_particles = merged_info['deactivated'].union(split_info['deactivated'])
        
        phase3_particles = [p for p in active_particles if p['id'] not in deactivated_particles]
        
        for particle in phase3_particles:
            # Check for spontaneous disappearance
            if np.random.random() < self.spontaneous_disappear_prob:
                self.record_particle_state(particle, EventLabel.NORMAL)
                particle['active'] = False
                particle['death_frame'] = self.current_frame
                continue
            
            # Update particle position
            left_boundaries = self.update_particle_position(particle, dt)
            
            # Handle boundary conditions for non-reflective boundaries
            if left_boundaries and not self.reflective_boundaries:
                self.record_particle_state(particle, EventLabel.NORMAL)
                particle['active'] = False
                particle['death_frame'] = self.current_frame
                continue
            
            # Record normal state
            self.record_particle_state(particle, EventLabel.NORMAL)
        
        # Handle spontaneous particle appearance
        if np.random.random() < self.spontaneous_appear_prob:
            x = np.random.uniform(self.x_range[0] + 10, self.x_range[1] - 10)
            y = np.random.uniform(self.y_range[0] + 10, self.y_range[1] - 10)
            z = np.random.uniform(self.z_range[0] + 10, self.z_range[1] - 10)
            mass = np.random.lognormal(np.log(150000), 0.5)
            mass = np.clip(mass, self.min_mass, self.max_mass)
            new_id = self.add_particle(x, y, z, mass)
            
            # Record initial state as NORMAL
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            self.record_particle_state(new_particle, EventLabel.NORMAL)

    def update(self, dt=0.1):
        """Main update function with fixed phase processing"""
        self.current_frame += 1
        self.current_frame_events = {'merges': [], 'splits': []}

        # Clean up old event particles from tracking
        self.recent_event_particles = {
            pid: frame for pid, frame in self.recent_event_particles.items()
            if self.current_frame - frame <= self.event_cooldown
        }

        # Refresh active particles list before each phase
        active_particles = [p for p in self.particles if p['active']]
        
        # PHASE 1: MERGE PROCESSING
        merged_info = self.process_merge_phase(active_particles)

        # Refresh again for phase 2
        active_particles = [p for p in self.particles if p['active']]
        
        # PHASE 2: SPLIT PROCESSING
        split_info = self.process_split_phase(active_particles, merged_info)

        # Refresh again for phase 3
        active_particles = [p for p in self.particles if p['active']]
        
        # PHASE 3: NORMAL UPDATE
        self.process_normal_update_phase(active_particles, merged_info, split_info, dt)

    def export_to_csv(self, filename):
        """Export simulation data to CSV"""
        data = []
        
        for particle in self.particles:
            death_frame = particle['death_frame'] if particle['death_frame'] is not None else self.current_frame
            duration = death_frame - particle['birth_frame']
            
            for i, (x, y, z) in enumerate(particle['trajectory']):
                frame = particle['birth_frame'] + i
                
                noisy_x = x + np.random.normal(0, self.position_noise_sigma)
                noisy_y = y + np.random.normal(0, self.position_noise_sigma)
                noisy_z = z + np.random.normal(0, self.position_noise_sigma)
                
                if i >= len(particle['mass_history']) or i >= len(particle['label_history']):
                    warnings.warn(f"Particle {particle['id']} missing data at index {i}")
                    continue
                    
                noisy_mass = particle['mass_history'][i] * np.random.normal(1, self.mass_noise_cv)
                event_label = particle['label_history'][i]

                data.append({
                    'frame': float(frame),
                    'x': noisy_x,
                    'y': noisy_y,
                    'z': noisy_z,
                    'mass': noisy_mass,
                    'particle': float(particle['id']),
                    'duration': float(duration),
                    'event_label': int(event_label),
                    'true_particle_id': float(particle['id']),
                    'parent_ids': ','.join(map(str, particle['parent_ids'])) if particle['parent_ids'] else '',
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(['frame', 'particle'])
        df.to_csv(filename, index=False)
        
        return df
    
def run_single_simulation(sim_id):
    np.random.seed(sim_id)
    
    output_dir = 'data/tracked_simdata_clean'
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = 200
    initial_particles = (13, 25)
    num_initial = np.random.randint(initial_particles[0], initial_particles[1] + 1)
    
    sim = TrackedParticleSimulator(
        x_range=(0, 100),
        y_range=(0, 100),
        z_range=(0, 50),  
        min_mass=50000,
        max_mass=500000,
        merge_distance_factor=10.0,
        split_mass_threshold=100000,
        merge_prob=0.8,
        split_prob=0.01,
        merge_linking_prob=0.8,
        split_linking_prob=0.8,
        spontaneous_appear_prob=0.00,
        spontaneous_disappear_prob=0.00,
        position_noise_sigma=0.5,
        mass_noise_cv=0.1,
        reflective_boundaries=True,
        event_cooldown=5
    )
    
    for i in range(num_initial):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        z = np.random.uniform(10, 40)
        mass = np.random.lognormal(np.log(150000), 0.5)
        mass = np.clip(mass, sim.min_mass, sim.max_mass)
        sim.add_particle(x, y, z, mass) 
    
    for frame in range(num_frames):
        sim.update(dt=0.1)
    
    output_file = os.path.join(output_dir, f'tracked_particles_3d_{sim_id:05d}.csv')
    df = sim.export_to_csv(output_file)
    
    return sim_id, len(df)

def run_multiprocess_simulations(num_simulations=10000, num_cores=7):
    print(f"Starting {num_simulations} 3D simulations using {num_cores} cores...")
    start_time = time.time()
    
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(run_single_simulation, range(num_simulations)),
            total=num_simulations,
            desc="3D Simulations"
        ))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    total_rows = sum(result[1] for result in results)
    print(f"\nAll 3D simulations completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per simulation: {total_time/num_simulations:.3f} seconds")
    print(f"Total data rows generated: {total_rows:,}")

if __name__ == "__main__":
    run_multiprocess_simulations(num_simulations=1, num_cores=1)