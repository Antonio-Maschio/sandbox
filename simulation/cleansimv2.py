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
        
        # if particle['id'] in self.recent_event_particles: #do not understand this 
        #     if self.current_frame - self.recent_event_particles[particle['id']] <= self.event_cooldown:
        #         return False
        
        if particle['parent_ids'] and self.current_frame - particle['birth_frame'] <= self.event_cooldown:
            return False
            
        return True
    
    def update_particle_position(self, particle, dt=0.1):
        """Update particle position with Brownian motion"""
        if not particle['active']: # is this needed?
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
                
            search_radius = self.merge_distance_factor * (p1['radius'] / self.pixel_size) ## maybe remove pixel size? for now is working fine
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
            if p1['id'] in merged_particles or p2['id'] in merged_particles: # if already merged, skip this cycle.
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

                print(f"Frame {self.current_frame}: Merged particles {p1['id']} and {p2['id']}") 
                       
        return merged_particles

    def _process_linking_merge(self, p1, p2, new_x, new_y, new_z, total_mass):
        """Process a linking merge where one particle continues"""
        if np.random.random()> 0.5:
            continuing_particle = p1
            disappearing_particle = p2
        else:
            continuing_particle = p2
            disappearing_particle = p1
        
        # Record pre-merge state for both particles
        self.record_particle_state(disappearing_particle, EventLabel.MERGE)
        self.record_particle_state(continuing_particle, EventLabel.MERGE)
        
        # Deactivate disappearing particle # update this to a function that removes particles.
        disappearing_particle['active'] = False
        disappearing_particle['death_frame'] = self.current_frame
        disappearing_particle['last_event_frame'] = self.current_frame
        
        # Create merged lineage
        merged_lineage = []

        merged_lineage.extend(continuing_particle['parent_ids'])
        # merged_lineage.append(continuing_particle['id']) # this will stay alive so no reasone to be in lineage

        merged_lineage.extend(disappearing_particle['parent_ids'])
        merged_lineage.append(disappearing_particle['id'])
        
        seen = set()

        merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))] #removes duplicates
        
        
        # Update continuing particle
        self.update_existing_particle(continuing_particle, new_x, new_y, new_z, 
                                    total_mass, merged_lineage)
        
        continuing_particle['last_event_frame'] = self.current_frame

        #####
        continuing_particle['event_info'] = ('merge', self.current_frame)

        # Record post-merge state
        self.record_particle_state(continuing_particle, EventLabel.POST_MERGE)
        
        # Update tracking
        self.recent_event_particles[continuing_particle['id']] = self.current_frame
        # self.recent_event_particles[disappearing_particle['id']] = self.current_frame
        
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
        new_id = self.add_particle(new_x, new_y, new_z, total_mass,birth_frame=self.current_frame+1, 
                                parent_ids=merged_lineage, inherit_cooldown=True) # t+1?`
        
        # Record POST_MERGE state for new particle
        new_particle = next(p for p in self.particles if p['id'] == new_id)
        self.record_particle_state(new_particle, EventLabel.POST_MERGE)
        
        # Update recent event tracking
        # self.recent_event_particles[p1['id']] = self.current_frame
        # self.recent_event_particles[p2['id']] = self.current_frame
        self.recent_event_particles[new_id] = self.current_frame
        
        # Record merge event
        self.current_frame_events['merges'].append({
            'parent_ids': [p1['id'], p2['id']],
            'child_id': new_id,
            'is_linking': False
        })