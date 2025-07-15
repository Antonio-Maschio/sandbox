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
"""clean simulation without any noise yet."""
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
        """Check if a particle can have an event based on cooldown"""
        if self.current_frame - particle.get('last_event_frame', -float('inf')) <= self.event_cooldown:
            return False
        
        if particle['id'] in self.recent_event_particles:
            if self.current_frame - self.recent_event_particles[particle['id']] <= self.event_cooldown:
                return False
        
        if particle['parent_ids'] and self.current_frame - particle['birth_frame'] <= self.event_cooldown:
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
    
    def find_merge_candidates(self, particles):
        """Find potential merge candidates using spatial proximity"""
        if len(particles) < 2:
            return []
            
        positions = np.array([[p['x'], p['y'], p['z']] for p in particles])
        tree = KDTree(positions)
        
        merge_pairs = []
        for i, p1 in enumerate(particles):
            if not self.can_have_event(p1):
                continue
                
            if p1['last_updated_frame'] != self.current_frame:
                continue
                
            search_radius = self.merge_distance_factor * (p1['radius'] / self.pixel_size)
            neighbors = tree.query_ball_point(positions[i], search_radius)
            
            for j in neighbors:
                if j <= i:
                    continue
                    
                p2 = particles[j]
                
                if not self.can_have_event(p2):
                    continue
                
                if p2['last_updated_frame'] != self.current_frame:
                    continue
                    
                if p1['mass'] + p2['mass'] > self.max_mass:
                    continue
                    
                dist = np.linalg.norm(positions[i] - positions[j])
                merge_threshold = self.merge_distance_factor * (p1['radius'] + p2['radius']) / (2 * self.pixel_size)
                
                if dist < merge_threshold:
                    if (p1['parent_ids'] and self.current_frame - p1['birth_frame'] <= self.event_cooldown) or \
                       (p2['parent_ids'] and self.current_frame - p2['birth_frame'] <= self.event_cooldown):
                        continue
                    
                    merge_pairs.append((p1, p2))
                    
        return merge_pairs

    def process_merge_phase(self, active_particles):
        """Phase 1: Process merge events"""
        merged_particles = set()
        
        merge_candidates = self.find_merge_candidates(active_particles)
        
        for p1, p2 in merge_candidates:
            if p1['id'] in merged_particles or p2['id'] in merged_particles:
                continue
            
            if np.random.random() < self.merge_prob:
                total_mass = p1['mass'] + p2['mass']
                new_x = (p1['x'] * p1['mass'] + p2['x'] * p2['mass']) / total_mass
                new_y = (p1['y'] * p1['mass'] + p2['y'] * p2['mass']) / total_mass
                new_z = (p1['z'] * p1['mass'] + p2['z'] * p2['mass']) / total_mass
                
                if np.random.random() < self.merge_linking_prob:
                    # LINKING MERGE
                    self._process_linking_merge(p1, p2, new_x, new_y, new_z, total_mass)
                else:
                    # NORMAL MERGE
                    self._process_normal_merge(p1, p2, new_x, new_y, new_z, total_mass)
                
                merged_particles.add(p1['id'])
                merged_particles.add(p2['id'])
        
        return merged_particles

    def _process_linking_merge(self, p1, p2, new_x, new_y, new_z, total_mass):
        """Process a linking merge where one particle continues"""
        if p1['mass'] >= p2['mass']:
            continuing_particle = p1
            disappearing_particle = p2
        else:
            continuing_particle = p2
            disappearing_particle = p1
        
        # Record pre-merge state for both particles
        self.record_particle_state(disappearing_particle, EventLabel.MERGE)
        self.record_particle_state(continuing_particle, EventLabel.MERGE)
        
        # Deactivate disappearing particle
        disappearing_particle['active'] = False
        disappearing_particle['death_frame'] = self.current_frame
        disappearing_particle['last_event_frame'] = self.current_frame
        
        # Create merged lineage
        merged_lineage = []
        merged_lineage.extend(continuing_particle['parent_ids'])
        merged_lineage.append(continuing_particle['id'])
        merged_lineage.extend(disappearing_particle['parent_ids'])
        merged_lineage.append(disappearing_particle['id'])
        seen = set()
        merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))]
        
        # Update continuing particle
        self.update_existing_particle(continuing_particle, new_x, new_y, new_z, 
                                    total_mass, merged_lineage)
        continuing_particle['last_event_frame'] = self.current_frame
        #####
        continuing_particle['event_info'] = ('merge', self.current_frame)
        # Record post-merge state
        self.record_particle_state(continuing_particle, EventLabel.NORMAL)
        
        # Update tracking
        self.recent_event_particles[continuing_particle['id']] = self.current_frame
        self.recent_event_particles[disappearing_particle['id']] = self.current_frame
        
        self.current_frame_events['merges'].append({
            'parent_ids': [disappearing_particle['id'], continuing_particle['id']],
            'child_id': continuing_particle['id'],
            'is_linking': True
        })

    def _process_normal_merge(self, p1, p2, new_x, new_y, new_z, total_mass):
        """Process a normal merge where both particles disappear and new one is created"""
        # Record states for both particles
        self.record_particle_state(p1, EventLabel.MERGE)
        self.record_particle_state(p2, EventLabel.MERGE)
        
        # Deactivate both particles
        p1['active'] = False
        p1['death_frame'] = self.current_frame
        p1['last_event_frame'] = self.current_frame
        
        p2['active'] = False
        p2['death_frame'] = self.current_frame
        p2['last_event_frame'] = self.current_frame
        
        # Create merged lineage
        merged_lineage = []
        merged_lineage.extend(p1['parent_ids'])
        merged_lineage.append(p1['id'])
        merged_lineage.extend(p2['parent_ids'])
        merged_lineage.append(p2['id'])
        # Remove duplicates while preserving order
        seen = set()
        merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))]
        
        # Create new particle from merge
        new_id = self.add_particle(new_x, new_y, new_z, total_mass, 
                                parent_ids=merged_lineage, inherit_cooldown=True)
        
        # Record POST_MERGE state for new particle
        new_particle = next(p for p in self.particles if p['id'] == new_id)
        self.record_particle_state(new_particle, EventLabel.POST_MERGE)
        
        # Update recent event tracking
        self.recent_event_particles[p1['id']] = self.current_frame
        self.recent_event_particles[p2['id']] = self.current_frame
        self.recent_event_particles[new_id] = self.current_frame
        
        # Record merge event
        self.current_frame_events['merges'].append({
            'parent_ids': [p1['id'], p2['id']],
            'child_id': new_id,
            'is_linking': False
        })

    def process_split_phase(self, active_particles, merged_particles):
        """Phase 2: Process split events"""
        split_particles = set()
        
        phase2_particles = [p for p in active_particles if p['id'] not in merged_particles]
        
        for particle in phase2_particles:
            if (np.random.random() < self.split_prob and 
                self.can_have_event(particle) and
                particle['mass'] >= self.split_mass_threshold):
                
                kinetic_factor = particle['D'] / np.mean([p['D'] for p in active_particles])
                if np.random.random() < kinetic_factor:
                    ratio = np.random.uniform(0.4, 0.6)
                    mass1 = particle['mass'] * ratio
                    mass2 = particle['mass'] * (1 - ratio)
                    
                    phi = np.random.uniform(0, 2 * np.pi)
                    theta = np.random.uniform(0, np.pi)
                    sep_dist = np.random.uniform(1, 12)
                    
                    dx = sep_dist * np.sin(theta) * np.cos(phi)
                    dy = sep_dist * np.sin(theta) * np.sin(phi)
                    dz = sep_dist * np.cos(theta)
                    
                    if np.random.random() < self.split_linking_prob:
                        self._process_linking_split(particle, mass1, mass2, dx, dy, dz)
                    else:
                        self._process_normal_split(particle, mass1, mass2, dx, dy, dz)
                    
                    split_particles.add(particle['id'])
        
        return split_particles

    def _process_linking_split(self, particle, mass1, mass2, dx, dy, dz):
        """Process a linking split where one particle continues"""
        self.record_particle_state(particle, EventLabel.SPLIT)
        
        # Determine which mass continues (larger one)
        if mass1 >= mass2:
            new_x = particle['x'] - dx/2
            new_y = particle['y'] - dy/2
            new_z = particle['z'] - dz/2
            continuing_mass = mass1
            
            # Create child lineage for new particle
            child_lineage = particle['parent_ids'].copy()
            child_lineage.append(particle['id'])
            child2_id = self.add_particle(
                particle['x'] + dx/2, particle['y'] + dy/2, particle['z'] + dz/2,
                mass2, parent_ids=child_lineage, inherit_cooldown=True
            )
            
            # Record POST_SPLIT state for new child
            child2 = next(p for p in self.particles if p['id'] == child2_id)
            self.record_particle_state(child2, EventLabel.POST_SPLIT)
            
            # Update event tracking
            self.recent_event_particles[particle['id']] = self.current_frame
            self.recent_event_particles[child2_id] = self.current_frame
            
            # Record split event
            self.current_frame_events['splits'].append({
                'parent_id': particle['id'],
                'child_ids': [particle['id'], child2_id],
                'is_linking': True
            })
        else:
            new_x = particle['x'] + dx/2
            new_y = particle['y'] + dy/2
            new_z = particle['z'] + dz/2
            continuing_mass = mass2
            
            # Create child lineage for new particle
            child_lineage = particle['parent_ids'].copy()
            child_lineage.append(particle['id'])
            child1_id = self.add_particle(
                particle['x'] - dx/2, particle['y'] - dy/2, particle['z'] - dz/2,
                mass1, parent_ids=child_lineage, inherit_cooldown=True
            )
            
            # Record POST_SPLIT state for new child
            child1 = next(p for p in self.particles if p['id'] == child1_id)
            self.record_particle_state(child1, EventLabel.POST_SPLIT)
            
            # Update event tracking
            self.recent_event_particles[particle['id']] = self.current_frame
            self.recent_event_particles[child1_id] = self.current_frame
            
            # Record split event
            self.current_frame_events['splits'].append({
                'parent_id': particle['id'],
                'child_ids': [child1_id, particle['id']],
                'is_linking': True
            })
        
        # Update continuing particle
        parent_lineage = particle['parent_ids'].copy()
        parent_lineage.append(particle['id'])
        self.update_existing_particle(particle, new_x, new_y, new_z, 
                                    continuing_mass, parent_lineage)
        particle['last_event_frame'] = self.current_frame
        
        # Record post split state after split
        self.record_particle_state(particle, EventLabel.POST_SPLIT)

    def _process_normal_split(self, particle, mass1, mass2, dx, dy, dz):
        """Process a normal split where particle disappears and two new ones are created"""
        self.record_particle_state(particle, EventLabel.SPLIT)
        
        # Deactivate parent particle
        particle['active'] = False
        particle['death_frame'] = self.current_frame
        particle['last_event_frame'] = self.current_frame
        
        # Create child lineage
        child_lineage = particle['parent_ids'].copy()
        child_lineage.append(particle['id'])
        
        # Create two new particles
        child1_id = self.add_particle(
            particle['x'] - dx/2, particle['y'] - dy/2, particle['z'] - dz/2,
            mass1, parent_ids=child_lineage, inherit_cooldown=True
        )
        child2_id = self.add_particle(
            particle['x'] + dx/2, particle['y'] + dy/2, particle['z'] + dz/2,
            mass2, parent_ids=child_lineage, inherit_cooldown=True
        )
        
        # Record POST_SPLIT state for both children
        child1 = next(p for p in self.particles if p['id'] == child1_id)
        child2 = next(p for p in self.particles if p['id'] == child2_id)
        self.record_particle_state(child1, EventLabel.POST_SPLIT)
        self.record_particle_state(child2, EventLabel.POST_SPLIT)
        
        # Update event tracking
        self.recent_event_particles[particle['id']] = self.current_frame
        self.recent_event_particles[child1_id] = self.current_frame
        self.recent_event_particles[child2_id] = self.current_frame
        
        # Record split event
        self.current_frame_events['splits'].append({
            'parent_id': particle['id'],
            'child_ids': [child1_id, child2_id],
            'is_linking': False
        })

    def process_normal_update_phase(self, active_particles, merged_particles, split_particles, dt=0.1):
        """
        Phase 3: Process normal updates for particles not involved in merge/split events
        
        Args:
            active_particles: List of active particles to process
            merged_particles: Set of particle IDs that were involved in merge events
            split_particles: Set of particle IDs that were involved in split events
            dt: Time step for position updates
        """
        # Only process particles that weren't involved in merges or splits
        phase3_particles = [p for p in active_particles 
                        if p['id'] not in merged_particles and p['id'] not in split_particles]
        
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
        
        # Handle false positive (ghost particle) generation  
        # if np.random.random() < self.false_positive_rate:
        #     self.add_ghost_particle()
        
        # Update ghost particles
        # self.ghost_particles = [g for g in self.ghost_particles 
        #                     if g['frames_lived'] < g['lifetime']]
        # for ghost in self.ghost_particles:
        #     self.update_ghost_particle(ghost)

    def update(self, dt=0.1):
        """Main update function with 3 phases only"""
        self.current_frame += 1
        self.current_frame_events = {'merges': [], 'splits': []}
        
        # Clean up old event particles from tracking
        self.recent_event_particles = {
            pid: frame for pid, frame in self.recent_event_particles.items()
            if self.current_frame - frame <= self.event_cooldown
        }
        
        active_particles = [p for p in self.particles if p['active']]
        
        # PHASE 1: MERGE PROCESSING
        merged_particles = self.process_merge_phase(active_particles)
        
        # PHASE 2: SPLIT PROCESSING
        split_particles = self.process_split_phase(active_particles, merged_particles)
        
        # PHASE 3: NORMAL UPDATE (including spontaneous events)
        self.process_normal_update_phase(active_particles, merged_particles, split_particles, dt)

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
        merge_prob=0.7,
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