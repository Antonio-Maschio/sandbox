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

class TrackedParticleSimulator:
    def __init__(self, 
                 x_range=(0, 200), y_range=(0, 200), z_range=(0, 200),
                 min_mass=50000, max_mass=500000,
                 temperature=300, viscosity=0.1, pixel_size=100,
                 merge_distance_factor=2.0, split_mass_threshold=100000,
                 merge_prob=0.7, split_prob=0.005,
                 merge_linking_prob=0.8, split_linking_prob=0.8,
                 spontaneous_appear_prob=0.002, spontaneous_disappear_prob=0.001,
                 position_noise_sigma=0.5, mass_noise_cv=0.1,
                 reflective_boundaries=True, event_cooldown=5,
                 enable_warnings=True, detection_prob=0.95, max_undetectable_frames=3):
        
        # Simulation boundaries
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
        self.merge_prob = merge_prob
        self.split_prob = split_prob
        self.merge_linking_prob = merge_linking_prob
        self.split_linking_prob = split_linking_prob
        self.spontaneous_appear_prob = spontaneous_appear_prob
        self.spontaneous_disappear_prob = spontaneous_disappear_prob
        
        # Noise parameters
        self.position_noise_sigma = position_noise_sigma
        self.mass_noise_cv = mass_noise_cv
        
        # Simulation settings
        self.reflective_boundaries = reflective_boundaries
        self.event_cooldown = event_cooldown
        self.enable_warnings = enable_warnings
        self.detection_prob = detection_prob
        self.max_undetectable_frames = max_undetectable_frames
        
        # State tracking
        self.particles = []
        self.next_id = 0
        self.current_frame = 0
        self.all_merges = []
        self.all_splits = []
        self.frame_warnings = []
        self.pending_post_events = []

    def log_warning(self, message):
        if self.enable_warnings:
            warning_msg = f"FRAME {self.current_frame}: {message}"
            print(f"WARNING: {warning_msg}")
            self.frame_warnings.append(warning_msg)

    def validate_particle_frame_consistency(self, particle, context=""):
        if not particle['active']:
            return True
            
        if particle['birth_frame'] > self.current_frame:
            self.log_warning(f"SYNC ERROR - {context} Particle {particle['id']} has future birth_frame")
            return False
            
        if particle['death_frame'] is not None and particle['death_frame'] < self.current_frame:
            self.log_warning(f"SYNC ERROR - {context} Active particle {particle['id']} has past death_frame")
            return False
            
        expected_recordings = self.current_frame - particle['birth_frame']
        actual_recordings = len(particle['trajectory'])
        
        if actual_recordings > expected_recordings + 1:
            self.log_warning(f"SYNC ERROR - {context} Particle {particle['id']} has too many recordings")
            return False
            
        return True

    def process_detection(self, particle):
        if not particle['active']:
            return
            
        if np.random.random() < particle['detection_prob']:
            if not particle['detectable']:
                if self.enable_warnings:
                    print(f"🔍 RE-DETECTION: Particle {particle['id']} after {particle['consecutive_undetectable']} frames")
            
            particle['detectable'] = True
            particle['last_detected_frame'] = self.current_frame
            particle['consecutive_undetectable'] = 0
        else:
            if particle['detectable'] and self.enable_warnings:
                print(f"👻 DETECTION LOST: Particle {particle['id']} at frame {self.current_frame}")
            
            particle['detectable'] = False
            particle['undetectable_frames'].append(self.current_frame)
            particle['consecutive_undetectable'] += 1
            
            if particle['consecutive_undetectable'] > self.max_undetectable_frames:
                if self.enable_warnings:
                    print(f"💀 TRACKING LOST: Particle {particle['id']} after {particle['consecutive_undetectable']} frames")
                particle['active'] = False
                particle['death_frame'] = self.current_frame

    def record_particle_state(self, particle, label):
        particle_id = particle['id']
        
        if not particle.get('detectable', True):
            if self.enable_warnings:
                print(f"👻 SKIP RECORD: Particle {particle_id} not detectable")
            return
        
        if particle.get('recorded_this_frame', False):
            self.log_warning(f"DOUBLE RECORD - Particle {particle_id} already recorded")
            return
            
        if not particle['active']:
            self.log_warning(f"INACTIVE RECORD - Recording inactive particle {particle_id}")
            return
            
        if not self.validate_particle_frame_consistency(particle, "RECORD"):
            return
            
        expected_frame = particle['birth_frame'] + len(particle['trajectory'])
        if expected_frame != self.current_frame:
            gap_size = self.current_frame - expected_frame
            if gap_size > 0 and self.enable_warnings:
                print(f"🔍 DETECTION GAP: Particle {particle_id} has {gap_size} frame gap")
            
        if particle['death_frame'] is not None and self.current_frame > particle['death_frame']:
            self.log_warning(f"POSTHUMOUS RECORD - Particle {particle_id} recorded after death")
            return
        
        if self.enable_warnings and label != EventLabel.NORMAL:
            print(f"DEBUG RECORD: Frame {self.current_frame} - Particle {particle_id} with {EventLabel(label).name}")
            
        particle['trajectory'].append((particle['x'], particle['y'], particle['z'], self.current_frame))
        particle['mass_history'].append(particle['mass'])
        particle['label_history'].append(label)
        particle['recorded_this_frame'] = True
        particle['last_recorded_frame'] = self.current_frame

    def add_particle(self, x, y, z, mass, parent_ids=None, birth_frame=None):
        if birth_frame is None:
            birth_frame = self.current_frame
        if parent_ids is None:
            parent_ids = []
            
        if birth_frame > self.current_frame + 1:
            self.log_warning(f"FUTURE BIRTH - Particle birth_frame too far in future")
        if birth_frame < 0:
            self.log_warning(f"NEGATIVE BIRTH - Negative birth_frame")
            birth_frame = 0
        if mass < self.min_mass or mass > self.max_mass:
            self.log_warning(f"MASS OUT OF BOUNDS - Mass {mass} outside bounds")
            
        for parent_id in parent_ids:
            parent_particle = next((p for p in self.particles if p['id'] == parent_id), None)
            if parent_particle is None:
                self.log_warning(f"INVALID PARENT - Parent ID {parent_id} not found")
            elif parent_particle['active'] and birth_frame <= self.current_frame:
                self.log_warning(f"ACTIVE PARENT - Parent {parent_id} still active for current/past birth")
        
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
            'recorded_this_frame': False, 'last_recorded_frame': -1,
            'detectable': True, 'detection_prob': self.detection_prob,
            'undetectable_frames': [], 'last_detected_frame': -1,
            'consecutive_undetectable': 0
        }
        
        self.particles.append(particle)
        self.next_id += 1
        return particle['id']
    
    def update_particle_properties(self, particle, x, y, z, mass, parent_ids=None):
        if not particle['active']:
            self.log_warning(f"INACTIVE UPDATE - Updating inactive particle {particle['id']}")
            return
            
        if parent_ids is not None:
            particle['parent_ids'] = parent_ids.copy()
            
        if mass < self.min_mass or mass > self.max_mass:
            self.log_warning(f"MASS UPDATE OUT OF BOUNDS - Mass {mass} outside bounds")
            
        volume = mass / 100000
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 100
        D = self.kB * self.T / (6 * np.pi * self.eta * radius * 1e-9)
        
        particle.update({'x': x, 'y': y, 'z': z, 'mass': mass, 'radius': radius,
                        'D': D * 1e18 / (self.pixel_size**2)})

    def can_have_event(self, particle):
        if not particle['active']:
            return False
        if self.current_frame - particle['last_event_frame'] <= self.event_cooldown:
            return False
        if particle['parent_ids'] and self.current_frame - particle['birth_frame'] <= self.event_cooldown:
            return False
        return True

    def update_particle_position(self, particle, dt=0.1):
        if not particle['active']:
            self.log_warning(f"POSITION UPDATE - Updating position of inactive particle {particle['id']}")
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
        return False
    
    def find_merge_candidates(self, particles):
        if len(particles) < 2:
            return []
            
        for particle in particles:
            self.validate_particle_frame_consistency(particle, "MERGE_CANDIDATE")
                
        positions = np.array([[p['x'], p['y'], p['z']] for p in particles])
        tree = KDTree(positions)
        merge_pairs = []
        
        for i, p1 in enumerate(particles):
            if not self.can_have_event(p1):
                continue

            search_radius = self.merge_distance_factor * (p1['radius'] / self.pixel_size)
            neighbors = tree.query_ball_point(positions[i], search_radius)
            
            for j in neighbors:
                if j <= i:
                    continue
                p2 = particles[j]
                if not self.can_have_event(p2):
                    continue
                if p1['mass'] + p2['mass'] > self.max_mass:
                    continue

                dist = np.linalg.norm(positions[i] - positions[j])
                merge_threshold = self.merge_distance_factor * (p1['radius'] + p2['radius']) / (2 * self.pixel_size)
                if dist < merge_threshold:
                    merge_pairs.append((p1, p2))

        return merge_pairs

    def process_merge(self, p1, p2):
        if not self.validate_particle_frame_consistency(p1, "MERGE_P1"):
            return
        if not self.validate_particle_frame_consistency(p2, "MERGE_P2"):
            return
            
        if not p1['active'] or not p2['active']:
            self.log_warning(f"MERGE INACTIVE - Attempting to merge inactive particles")
            return
            
        total_mass = p1['mass'] + p2['mass']
        if total_mass <= 0:
            self.log_warning(f"MERGE ZERO MASS - Invalid mass conservation")
            return
            
        new_x = (p1['x'] * p1['mass'] + p2['x'] * p2['mass']) / total_mass
        new_y = (p1['y'] * p1['mass'] + p2['y'] * p2['mass']) / total_mass
        new_z = (p1['z'] * p1['mass'] + p2['z'] * p2['mass']) / total_mass
        
        if self.enable_warnings:
            print(f"DEBUG: Frame {self.current_frame} - Processing merge between particles {p1['id']} and {p2['id']}")
        
        if np.random.random() < self.merge_linking_prob:
            # Linking merge
            continuing_particle = p1 if np.random.random() < 0.5 else p2
            disappearing_particle = p2 if continuing_particle == p1 else p1
            
            if self.enable_warnings:
                print(f"🔗 LINKING MERGE: {disappearing_particle['id']} disappears, {continuing_particle['id']} continues")
            
            self.record_particle_state(disappearing_particle, EventLabel.MERGE)
            self.record_particle_state(continuing_particle, EventLabel.MERGE)
            
            disappearing_particle['active'] = False
            disappearing_particle['death_frame'] = self.current_frame
            
            merged_lineage = []
            merged_lineage.extend(continuing_particle['parent_ids'])
            merged_lineage.extend(disappearing_particle['parent_ids'])
            merged_lineage.append(disappearing_particle['id'])
            seen = set()
            merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))]
            
            self.update_particle_properties(continuing_particle, new_x, new_y, new_z, 
                                          total_mass, merged_lineage)
            continuing_particle['last_event_frame'] = self.current_frame
            
            self.pending_post_events.append({
                'type': 'POST_MERGE',
                'particle_ids': [continuing_particle['id']]
            })
            
            self.all_merges.append({
                'parent_ids': [p1['id'], p2['id']],
                'child_id': continuing_particle['id'],
                'is_linking': True,
                'frame': self.current_frame
            })
            
        else:
            # Normal merge
            if self.enable_warnings:
                print(f"⚫ NORMAL MERGE: {p1['id']} and {p2['id']} disappear, creating new particle")
            
            self.record_particle_state(p1, EventLabel.MERGE)
            self.record_particle_state(p2, EventLabel.MERGE)
            
            p1['active'] = False
            p1['death_frame'] = self.current_frame
            p2['active'] = False
            p2['death_frame'] = self.current_frame
            
            merged_lineage = []
            merged_lineage.extend(p1['parent_ids'])
            merged_lineage.append(p1['id'])
            merged_lineage.extend(p2['parent_ids'])
            merged_lineage.append(p2['id'])
            seen = set()
            merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))]
            
            new_id = self.add_particle(new_x, new_y, new_z, total_mass, 
                                     parent_ids=merged_lineage, birth_frame=self.current_frame + 1)
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            new_particle['last_event_frame'] = self.current_frame
            
            if self.enable_warnings:
                print(f"   Created: Particle {new_id}")
            
            self.pending_post_events.append({
                'type': 'POST_MERGE',
                'particle_ids': [new_id]
            })
            
            self.all_merges.append({
                'parent_ids': [p1['id'], p2['id']],
                'child_id': new_id,
                'is_linking': False,
                'frame': self.current_frame
            })

    def process_split(self, particle):
        if not self.validate_particle_frame_consistency(particle, "SPLIT"):
            return
            
        if not particle['active']:
            self.log_warning(f"SPLIT INACTIVE - Attempting to split inactive particle {particle['id']}")
            return
            
        if particle['mass'] < self.split_mass_threshold:
            self.log_warning(f"SPLIT MASS TOO LOW - Particle {particle['id']} mass below threshold")
            return
            
        ratio = np.random.uniform(0.4, 0.6)
        mass1 = particle['mass'] * ratio
        mass2 = particle['mass'] * (1 - ratio)
        
        if abs((mass1 + mass2) - particle['mass']) > 1e-6:
            self.log_warning(f"SPLIT MASS CONSERVATION - Mass conservation error")
            
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        sep_dist = np.random.uniform(1, 12)
        
        dx = sep_dist * np.sin(theta) * np.cos(phi)
        dy = sep_dist * np.sin(theta) * np.sin(phi)
        dz = sep_dist * np.cos(theta)
        
        if self.enable_warnings:
            print(f"DEBUG: Frame {self.current_frame} - Processing split of particle {particle['id']}")
        
        if np.random.random() < self.split_linking_prob:
            # Linking split
            if mass1 >= mass2:
                continuing_mass, new_mass = mass1, mass2
                continuing_x = particle['x'] - dx/2
                continuing_y = particle['y'] - dy/2
                continuing_z = particle['z'] - dz/2
                new_x = particle['x'] + dx/2
                new_y = particle['y'] + dy/2
                new_z = particle['z'] + dz/2
            else:
                continuing_mass, new_mass = mass2, mass1
                continuing_x = particle['x'] + dx/2
                continuing_y = particle['y'] + dy/2
                continuing_z = particle['z'] + dz/2
                new_x = particle['x'] - dx/2
                new_y = particle['y'] - dy/2
                new_z = particle['z'] - dz/2
            
            if self.enable_warnings:
                print(f"🔗 LINKING SPLIT: Particle {particle['id']} continues, creating new particle")
            
            self.record_particle_state(particle, EventLabel.SPLIT)
            
            self.update_particle_properties(particle, continuing_x, continuing_y, continuing_z, 
                                          continuing_mass)
            particle['last_event_frame'] = self.current_frame
            
            child_lineage = particle['parent_ids'].copy()
            new_id = self.add_particle(new_x, new_y, new_z, new_mass, 
                                     parent_ids=child_lineage, birth_frame=self.current_frame + 1)
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            new_particle['last_event_frame'] = self.current_frame
            
            if self.enable_warnings:
                print(f"   Created: Particle {new_id}")
            
            self.pending_post_events.append({
                'type': 'POST_SPLIT',
                'particle_ids': [particle['id'], new_id]
            })
            
            child_ids = [particle['id'], new_id] if mass1 >= mass2 else [new_id, particle['id']]
            self.all_splits.append({
                'parent_id': particle['id'],
                'child_ids': child_ids,
                'is_linking': True,
                'frame': self.current_frame
            })
            
        else:
            # Normal split
            if self.enable_warnings:
                print(f"⚫ NORMAL SPLIT: Particle {particle['id']} disappears, creating two particles")
            
            self.record_particle_state(particle, EventLabel.SPLIT)
            particle['active'] = False
            particle['death_frame'] = self.current_frame
            
            child_lineage = particle['parent_ids'].copy()
            child_lineage.append(particle['id'])
            
            child1_id = self.add_particle(
                particle['x'] - dx/2, particle['y'] - dy/2, particle['z'] - dz/2,
                mass1, parent_ids=child_lineage.copy(), birth_frame=self.current_frame + 1
            )
            child2_id = self.add_particle(
                particle['x'] + dx/2, particle['y'] + dy/2, particle['z'] + dz/2,
                mass2, parent_ids=child_lineage.copy(), birth_frame=self.current_frame + 1
            )
            
            child1 = next(p for p in self.particles if p['id'] == child1_id)
            child2 = next(p for p in self.particles if p['id'] == child2_id)
            
            child1['last_event_frame'] = self.current_frame
            child2['last_event_frame'] = self.current_frame
            
            if self.enable_warnings:
                print(f"   Created: Particles {child1_id} and {child2_id}")
            
            self.pending_post_events.append({
                'type': 'POST_SPLIT',
                'particle_ids': [child1_id, child2_id]
            })
            
            self.all_splits.append({
                'parent_id': particle['id'],
                'child_ids': [child1_id, child2_id],
                'is_linking': False,
                'frame': self.current_frame
            })

    def process_pending_post_events(self):
        # Activate particles born this frame
        for particle in self.particles:
            if not particle['active'] and particle['birth_frame'] == self.current_frame:
                particle['active'] = True
                if self.enable_warnings:
                    print(f"🟢 BIRTH: Activating particle {particle['id']} at frame {self.current_frame}")
        
        if not self.pending_post_events:
            return
            
        if self.enable_warnings:
            print(f"📋 POST-EVENT PROCESSING - Frame {self.current_frame}: {len(self.pending_post_events)} events")
        
        for event in self.pending_post_events:
            event_type = event['type']
            particle_ids = event['particle_ids']
            
            for particle_id in particle_ids:
                particle = next((p for p in self.particles if p['id'] == particle_id), None)
                if particle and particle['active']:
                    if event_type == 'POST_MERGE':
                        self.record_particle_state(particle, EventLabel.POST_MERGE)
                    elif event_type == 'POST_SPLIT':
                        self.record_particle_state(particle, EventLabel.POST_SPLIT)
                elif self.enable_warnings:
                    print(f"     ❌ Particle {particle_id} - not found or inactive")
        
        self.pending_post_events = []

    def validate_label_sequence(self):
        """Validate label sequences - simplified for two-phase system"""
        for particle in self.particles:
            if not particle['label_history']:
                continue
                
            labels = particle['label_history']
            
            for i in range(len(labels) - 1):
                current_label = labels[i]
                next_label = labels[i + 1]
                
                if current_label == EventLabel.SPLIT and next_label == EventLabel.POST_MERGE:
                    self.log_warning(f"INVALID SEQUENCE - Particle {particle['id']} has SPLIT followed by POST_MERGE")
                
                if current_label == EventLabel.MERGE and next_label == EventLabel.POST_SPLIT:
                    self.log_warning(f"INVALID SEQUENCE - Particle {particle['id']} has MERGE followed by POST_SPLIT")

    def validate_event_labeling(self):
        """Validate event labeling - disabled for two-phase system"""
        pass

    def validate_frame_state(self):
        """Validate frame state - temporarily disabled"""
        pass

    def update(self, dt=0.1):
        self.current_frame += 1
        
        active_particles = [p for p in self.particles if p['active']]
        if not active_particles:
            self.log_warning("NO ACTIVE PARTICLES")
            return
            
        # Reset recording flags
        for particle in self.particles:
            particle['recorded_this_frame'] = False
        
        # Process pending POST events from previous frame
        self.process_pending_post_events()
        
        # Process detection for all active particles
        detection_status = []
        for particle in active_particles:
            old_detectable = particle.get('detectable', True)
            self.process_detection(particle)
            if self.enable_warnings:
                new_detectable = particle.get('detectable', True)
                if old_detectable != new_detectable:
                    status = "DETECTED" if new_detectable else "LOST"
                    detection_status.append(f"Particle {particle['id']}: {status}")
        
        if self.enable_warnings and detection_status:
            print(f"🔍 DETECTION CHANGES Frame {self.current_frame}: {', '.join(detection_status)}")
        
        unrecorded_particles = [p for p in active_particles if not p.get('recorded_this_frame', False)]
        processed_particles = set()
        
        # Process merge events
        merge_candidates = self.find_merge_candidates(unrecorded_particles)
        for p1, p2 in merge_candidates:
            if p1['id'] in processed_particles or p2['id'] in processed_particles:
                continue
            if np.random.random() < self.merge_prob:
                self.process_merge(p1, p2)
                processed_particles.add(p1['id'])
                processed_particles.add(p2['id'])
        
        # Process split events
        current_unrecorded = [p for p in self.particles if p['active'] and not p.get('recorded_this_frame', False)]
        for particle in current_unrecorded:
            if particle['id'] in processed_particles:
                continue
            if (np.random.random() < self.split_prob and 
                self.can_have_event(particle) and
                particle['mass'] >= self.split_mass_threshold):
                
                active_D_values = [p['D'] for p in current_unrecorded if p['active']]
                if active_D_values:
                    kinetic_factor = particle['D'] / np.mean(active_D_values)
                    if np.random.random() < kinetic_factor:
                        self.process_split(particle)
                        processed_particles.add(particle['id'])
        
        # Process normal updates
        final_unrecorded = [p for p in self.particles if p['active'] and not p.get('recorded_this_frame', False)]
        for particle in final_unrecorded:
            if particle['id'] in processed_particles:
                continue
                
            if np.random.random() < self.spontaneous_disappear_prob:
                self.record_particle_state(particle, EventLabel.NORMAL)
                particle['active'] = False
                particle['death_frame'] = self.current_frame
                continue
            
            left_boundaries = self.update_particle_position(particle, dt)
            if left_boundaries and not self.reflective_boundaries:
                self.record_particle_state(particle, EventLabel.NORMAL)
                particle['active'] = False
                particle['death_frame'] = self.current_frame
                continue
            
            self.record_particle_state(particle, EventLabel.NORMAL)
        
        # Handle spontaneous particle appearance
        if np.random.random() < self.spontaneous_appear_prob:
            x = np.random.uniform(self.x_range[0] + 10, self.x_range[1] - 10)
            y = np.random.uniform(self.y_range[0] + 10, self.y_range[1] - 10)
            z = np.random.uniform(self.z_range[0] + 10, self.z_range[1] - 10)
            mass = np.random.lognormal(np.log(150000), 0.5)
            mass = np.clip(mass, self.min_mass, self.max_mass)
            new_id = self.add_particle(x, y, z, mass)
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            self.record_particle_state(new_particle, EventLabel.NORMAL)

    def export_to_csv(self, filename):
        data = []
        export_warnings = []
        
        for particle in self.particles:
            if not particle['trajectory']:
                export_warnings.append(f"Particle {particle['id']} has empty trajectory")
                continue
                
            death_frame = particle['death_frame'] if particle['death_frame'] is not None else self.current_frame
            
            for i, trajectory_entry in enumerate(particle['trajectory']):
                if len(trajectory_entry) == 4:
                    x, y, z, frame = trajectory_entry
                else:
                    x, y, z = trajectory_entry
                    frame = particle['birth_frame'] + i
                
                if frame > death_frame or frame > self.current_frame:
                    export_warnings.append(f"Particle {particle['id']} invalid frame {frame}")
                    break
                
                total_duration = death_frame - particle['birth_frame'] + 1
                detected_duration = len(particle['trajectory'])
                
                noisy_x = x + np.random.normal(0, self.position_noise_sigma)
                noisy_y = y + np.random.normal(0, self.position_noise_sigma)
                noisy_z = z + np.random.normal(0, self.position_noise_sigma)
                
                if i >= len(particle['mass_history']) or i >= len(particle['label_history']):
                    export_warnings.append(f"Particle {particle['id']} missing data at index {i}")
                    continue
                    
                noisy_mass = particle['mass_history'][i] * np.random.normal(1, self.mass_noise_cv)
                event_label = particle['label_history'][i]

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
                    'undetectable_frames': ','.join(map(str, particle.get('undetectable_frames', []))),
                })
        
        if export_warnings and self.enable_warnings:
            print(f"\n=== EXPORT WARNINGS ({len(export_warnings)}) ===")
            for warning in export_warnings[:10]:
                print(f"EXPORT WARNING: {warning}")
            if len(export_warnings) > 10:
                print(f"... and {len(export_warnings) - 10} more warnings")
            print("=====================================\n")
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(['frame', 'particle'])
        else:
            print("ERROR: No data to export!")
            
        df.to_csv(filename, index=False)
        
        # Detection statistics
        if not df.empty and self.enable_warnings:
            detection_stats = df.groupby('particle').agg({
                'duration': 'first',
                'detected_duration': 'first'
            }).reset_index()
            detection_stats['detection_rate'] = detection_stats['detected_duration'] / detection_stats['duration']
            
            print(f"\n=== DETECTION STATISTICS ===")
            print(f"Average detection rate: {detection_stats['detection_rate'].mean():.3f}")
            particles_with_gaps = detection_stats[detection_stats['detection_rate'] < 1.0]
            print(f"Particles with gaps: {len(particles_with_gaps)}/{len(detection_stats)}")
            if len(particles_with_gaps) > 0:
                print(f"Worst detection rate: {particles_with_gaps['detection_rate'].min():.3f}")
            print("===========================\n")
        
        return df
    
    def print_summary(self):
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Total frames: {self.current_frame}")
        print(f"Total particles created: {self.next_id}")
        print(f"Total merges: {len(self.all_merges)}")
        print(f"Total splits: {len(self.all_splits)}")
        print(f"Total warnings: {len(self.frame_warnings)}")
        
        # Event mechanics examples
        print(f"\n=== EVENT MECHANICS EXAMPLES ===")
        
        normal_merges = [m for m in self.all_merges if not m['is_linking']]
        if normal_merges:
            example = normal_merges[0]
            print(f"⚫ NORMAL MERGE (Frame {example['frame']}): {example['parent_ids']} → {example['child_id']}")
        
        linking_merges = [m for m in self.all_merges if m['is_linking']]
        if linking_merges:
            example = linking_merges[0]
            print(f"🔗 LINKING MERGE (Frame {example['frame']}): {example['parent_ids']} → {example['child_id']}")
        
        normal_splits = [s for s in self.all_splits if not s['is_linking']]
        if normal_splits:
            example = normal_splits[0]
            print(f"⚫ NORMAL SPLIT (Frame {example['frame']}): {example['parent_id']} → {example['child_ids']}")
        
        linking_splits = [s for s in self.all_splits if s['is_linking']]
        if linking_splits:
            example = linking_splits[0]
            print(f"🔗 LINKING SPLIT (Frame {example['frame']}): {example['parent_id']} → {example['child_ids']}")
        
        if self.frame_warnings and self.enable_warnings:
            print(f"\nFirst 10 warnings:")
            for i, warning in enumerate(self.frame_warnings[:10]):
                print(f"  {i+1}. {warning}")
            if len(self.frame_warnings) > 10:
                print(f"  ... and {len(self.frame_warnings) - 10} more")
        
        print("=========================\n")

def run_single_simulation(sim_id):
    np.random.seed(sim_id)
    
    output_dir = 'data/tracked_simdata_dirty'
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = 200
    initial_particles = (13, 25)
    num_initial = np.random.randint(initial_particles[0], initial_particles[1] + 1)
    
    sim = TrackedParticleSimulator(
        x_range=(0, 100), y_range=(0, 100), z_range=(0, 50),  
        min_mass=50000, max_mass=500000,
        merge_distance_factor=10.0, split_mass_threshold=100000,
        merge_prob=0.8, split_prob=0.01,
        merge_linking_prob=0.8, split_linking_prob=0.8,
        spontaneous_appear_prob=0.00, spontaneous_disappear_prob=0.00,
        position_noise_sigma=0.5, mass_noise_cv=0.1,
        reflective_boundaries=True, event_cooldown=5,
        enable_warnings=False, detection_prob=0.90, max_undetectable_frames=3
    )
    
    # Initialize particles
    for i in range(num_initial):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        z = np.random.uniform(10, 40)
        mass = np.random.lognormal(np.log(150000), 0.5)
        mass = np.clip(mass, sim.min_mass, sim.max_mass)
        particle_id = sim.add_particle(x, y, z, mass)
        
        particle = next(p for p in sim.particles if p['id'] == particle_id)
        sim.record_particle_state(particle, EventLabel.NORMAL)
    
    # Run simulation
    for frame in range(num_frames):
        sim.update(dt=0.1)
    
    # Export data
    output_file = os.path.join(output_dir, f'tracked_particles_3d_{sim_id:05d}.csv')
    df = sim.export_to_csv(output_file)
    
    if sim_id == 0:
        sim.print_summary()
    
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
    run_multiprocess_simulations(num_simulations=1000, num_cores=7)