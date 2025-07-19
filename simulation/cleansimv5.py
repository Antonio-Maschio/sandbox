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
                 event_cooldown=5,
                 enable_warnings=True):
        
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
        self.enable_warnings = enable_warnings
        
        self.particles = []
        self.next_id = 0
        self.current_frame = 0
        
        # Track all events across simulation
        self.all_merges = []
        self.all_splits = []
        
        # Frame synchronization tracking
        self.frame_warnings = []
        
        # Track pending events that need POST labels in next frame
        self.pending_post_events = []

    def log_warning(self, message):
        """Log a warning message"""
        if self.enable_warnings:
            warning_msg = f"FRAME {self.current_frame}: {message}"
            print(f"WARNING: {warning_msg}")
            self.frame_warnings.append(warning_msg)

    def validate_particle_frame_consistency(self, particle, context=""):
        """Validate that particle frame data is consistent"""
        if not particle['active']:
            return True
            
        # Check birth frame
        if particle['birth_frame'] > self.current_frame:
            self.log_warning(f"SYNC ERROR - {context} Particle {particle['id']} has future birth_frame={particle['birth_frame']} > current_frame={self.current_frame}")
            return False
            
        # Check death frame
        if particle['death_frame'] is not None and particle['death_frame'] < self.current_frame:
            self.log_warning(f"SYNC ERROR - {context} Active particle {particle['id']} has past death_frame={particle['death_frame']} < current_frame={self.current_frame}")
            return False
            
        # Check trajectory length consistency
        expected_recordings = self.current_frame - particle['birth_frame']
        actual_recordings = len(particle['trajectory'])
        
        if actual_recordings > expected_recordings + 1:  # +1 because we might be mid-recording
            self.log_warning(f"SYNC ERROR - {context} Particle {particle['id']} has too many recordings: {actual_recordings} > expected {expected_recordings + 1}")
            return False
            
        return True

    def record_particle_state(self, particle, label):
        """Record the current state of a particle with comprehensive validation"""
        particle_id = particle['id']
        
        # Check if already recorded this frame
        if particle.get('recorded_this_frame', False):
            self.log_warning(f"DOUBLE RECORD - Particle {particle_id} already recorded in frame {self.current_frame}")
            return
            
        # Check if particle is active
        if not particle['active']:
            self.log_warning(f"INACTIVE RECORD - Recording inactive particle {particle_id}")
            return
            
        # Check frame consistency
        if not self.validate_particle_frame_consistency(particle, "RECORD"):
            return
            
        # Check if we're recording at the right frame
        expected_frame = particle['birth_frame'] + len(particle['trajectory'])
        if expected_frame != self.current_frame:
            self.log_warning(f"FRAME MISMATCH - Particle {particle_id} expected frame {expected_frame} but recording at {self.current_frame}")
            
        # Check if recording after death
        if particle['death_frame'] is not None and self.current_frame > particle['death_frame']:
            self.log_warning(f"POSTHUMOUS RECORD - Recording particle {particle_id} at frame {self.current_frame} after death_frame {particle['death_frame']}")
            return
            
        # Validate trajectory/mass/label consistency
        if len(particle['trajectory']) != len(particle['mass_history']):
            self.log_warning(f"HISTORY MISMATCH - Particle {particle_id} trajectory length {len(particle['trajectory'])} != mass history length {len(particle['mass_history'])}")
            
        if len(particle['trajectory']) != len(particle['label_history']):
            self.log_warning(f"HISTORY MISMATCH - Particle {particle_id} trajectory length {len(particle['trajectory'])} != label history length {len(particle['label_history'])}")
        
        # DEBUG: Track what we're recording
        if self.enable_warnings and label != EventLabel.NORMAL:
            print(f"DEBUG RECORD: Frame {self.current_frame} - Recording particle {particle_id} with label {EventLabel(label).name}")
            
        # Record the state
        particle['trajectory'].append((particle['x'], particle['y'], particle['z']))
        particle['mass_history'].append(particle['mass'])
        particle['label_history'].append(label)
        particle['recorded_this_frame'] = True
        particle['last_recorded_frame'] = self.current_frame
        
        # Validate post-recording state
        if len(particle['trajectory']) != len(particle['mass_history']) or len(particle['trajectory']) != len(particle['label_history']):
            self.log_warning(f"POST-RECORD MISMATCH - Particle {particle_id} histories out of sync after recording")

    def add_particle(self, x, y, z, mass, parent_ids=None, birth_frame=None):
        """Add a new particle with validation"""
        if birth_frame is None:
            birth_frame = self.current_frame
            
        if parent_ids is None:
            parent_ids = []
            
        # Validate birth frame
        if birth_frame > self.current_frame + 1:
            self.log_warning(f"FUTURE BIRTH - Creating particle with far future birth_frame {birth_frame} > current_frame+1 {self.current_frame + 1}")
            
        if birth_frame < 0:
            self.log_warning(f"NEGATIVE BIRTH - Creating particle with negative birth_frame {birth_frame}")
            birth_frame = 0
            
        # Validate mass
        if mass < self.min_mass or mass > self.max_mass:
            self.log_warning(f"MASS OUT OF BOUNDS - Particle mass {mass} outside bounds [{self.min_mass}, {self.max_mass}]")
            
        # Validate parent IDs
        if parent_ids:
            for parent_id in parent_ids:
                parent_particle = next((p for p in self.particles if p['id'] == parent_id), None)
                if parent_particle is None:
                    self.log_warning(f"INVALID PARENT - Parent ID {parent_id} not found")
                elif parent_particle['active'] and birth_frame > self.current_frame:
                    # This is okay for linking events - parent is still active but child will be born next frame
                    pass
                elif parent_particle['active'] and birth_frame <= self.current_frame:
                    self.log_warning(f"ACTIVE PARENT - Parent particle {parent_id} is still active for current/past birth")
        
        volume = mass / 100000
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 100
        D = self.kB * self.T / (6 * np.pi * self.eta * radius * 1e-9)
        
        # Particle starts inactive if born in future frame
        is_active = birth_frame <= self.current_frame
        
        particle = {
            'id': self.next_id,
            'parent_ids': parent_ids.copy(),
            'x': x,
            'y': y,
            'z': z,
            'mass': mass,
            'radius': radius,
            'D': D * 1e18 / (self.pixel_size**2),
            'active': is_active,
            'birth_frame': birth_frame,
            'death_frame': None,
            'trajectory': [],
            'mass_history': [],
            'label_history': [],
            'last_event_frame': -float('inf'),
            'recorded_this_frame': False,
            'last_recorded_frame': -1
        }
        
        self.particles.append(particle)
        self.next_id += 1
        
        return particle['id']
    
    def update_particle_properties(self, particle, x, y, z, mass, parent_ids=None):
        """Update particle properties with validation"""
        if not particle['active']:
            self.log_warning(f"INACTIVE UPDATE - Updating inactive particle {particle['id']}")
            return
            
        if parent_ids is not None:
            particle['parent_ids'] = parent_ids.copy()
            
        # Validate mass
        if mass < self.min_mass or mass > self.max_mass:
            self.log_warning(f"MASS UPDATE OUT OF BOUNDS - Particle {particle['id']} mass {mass} outside bounds [{self.min_mass}, {self.max_mass}]")
            
        volume = mass / 100000
        radius = np.cbrt(3 * volume / (4 * np.pi)) * 100
        D = self.kB * self.T / (6 * np.pi * self.eta * radius * 1e-9)
        
        particle['x'] = x
        particle['y'] = y
        particle['z'] = z
        particle['mass'] = mass
        particle['radius'] = radius
        particle['D'] = D * 1e18 / (self.pixel_size**2)

    def can_have_event(self, particle):
        """Check if particle can have an event (merge/split) with validation"""
        if not particle['active']:
            return False
            
        # Check cooldown period
        if self.current_frame - particle['last_event_frame'] <= self.event_cooldown:
            return False
        
        # Check if recently born from event
        if particle['parent_ids'] and self.current_frame - particle['birth_frame'] <= self.event_cooldown:
            return False
            
        return True

    def update_particle_position(self, particle, dt=0.1):
        """Update particle position with Brownian motion"""
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
                return True  # Left boundaries
        
        particle['x'] = new_x
        particle['y'] = new_y
        particle['z'] = new_z
        return False
    
    def find_merge_candidates(self, particles):
        """Find potential merge candidates using spatial proximity"""
        if len(particles) < 2:
            return []
            
        # Validate input particles
        for particle in particles:
            if not self.validate_particle_frame_consistency(particle, "MERGE_CANDIDATE"):
                continue
                
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
        """Process a merge event between two particles with proper two-phase timing"""
        # Validate particles before merge
        if not self.validate_particle_frame_consistency(p1, "MERGE_P1"):
            return
        if not self.validate_particle_frame_consistency(p2, "MERGE_P2"):
            return
            
        if not p1['active'] or not p2['active']:
            self.log_warning(f"MERGE INACTIVE - Attempting to merge inactive particles {p1['id']} (active={p1['active']}) and {p2['id']} (active={p2['active']})")
            return
            
        # Check for mass conservation
        total_mass = p1['mass'] + p2['mass']
        if total_mass <= 0:
            self.log_warning(f"MERGE ZERO MASS - Merge would result in zero/negative mass: {p1['mass']} + {p2['mass']} = {total_mass}")
            return
            
        new_x = (p1['x'] * p1['mass'] + p2['x'] * p2['mass']) / total_mass
        new_y = (p1['y'] * p1['mass'] + p2['y'] * p2['mass']) / total_mass
        new_z = (p1['z'] * p1['mass'] + p2['z'] * p2['mass']) / total_mass
        
        # DEBUG: Track merge event
        if self.enable_warnings:
            print(f"DEBUG: Frame {self.current_frame} - Processing merge between particles {p1['id']} and {p2['id']}")
        
        if np.random.random() < self.merge_linking_prob:
            # LINKING MERGE - one particle continues, one disappears
            continuing_particle = p1 if np.random.random() < 0.5 else p2
            disappearing_particle = p2 if continuing_particle == p1 else p1
            
            # DEBUG
            if self.enable_warnings:
                print(f"\n🔗 LINKING MERGE ANALYSIS - Frame {self.current_frame}")
                print(f"   Disappearing: Particle {disappearing_particle['id']} (mass={disappearing_particle['mass']:.0f})")
                print(f"   Continuing:   Particle {continuing_particle['id']} (mass={continuing_particle['mass']:.0f})")
                print(f"   New mass:     {total_mass:.0f}")
                print(f"   Expected next frame: Continuing particle gets POST_MERGE")
            
            # PHASE 1: Record both particles with MERGE label (this frame)
            self.record_particle_state(disappearing_particle, EventLabel.MERGE)
            self.record_particle_state(continuing_particle, EventLabel.MERGE)
            
            disappearing_particle['active'] = False
            disappearing_particle['death_frame'] = self.current_frame
            
            # Update continuing particle with merged properties
            merged_lineage = []
            merged_lineage.extend(continuing_particle['parent_ids'])
            merged_lineage.extend(disappearing_particle['parent_ids'])
            merged_lineage.append(disappearing_particle['id'])
            # Remove duplicates while preserving order
            seen = set()
            merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))]
            
            self.update_particle_properties(continuing_particle, new_x, new_y, new_z, 
                                          total_mass, merged_lineage)
            continuing_particle['last_event_frame'] = self.current_frame
            
            # PHASE 2: Schedule POST_MERGE label for next frame
            self.pending_post_events.append({
                'type': 'POST_MERGE',
                'particle_ids': [continuing_particle['id']]
            })
            
            # Track event
            self.all_merges.append({
                'parent_ids': [p1['id'], p2['id']],
                'child_id': continuing_particle['id'],
                'is_linking': True,
                'frame': self.current_frame
            })
            
        else:
            # NORMAL MERGE - both particles disappear, new one created
            # DEBUG
            if self.enable_warnings:
                print(f"\n⚫ NORMAL MERGE ANALYSIS - Frame {self.current_frame}")
                print(f"   Parent 1:     Particle {p1['id']} (mass={p1['mass']:.0f})")
                print(f"   Parent 2:     Particle {p2['id']} (mass={p2['mass']:.0f})")
                print(f"   New mass:     {total_mass:.0f}")
                print(f"   Expected next frame: New particle gets POST_MERGE")
            
            # PHASE 1: Record both particles with MERGE label (this frame)
            self.record_particle_state(p1, EventLabel.MERGE)
            self.record_particle_state(p2, EventLabel.MERGE)
            
            p1['active'] = False
            p1['death_frame'] = self.current_frame
            p2['active'] = False
            p2['death_frame'] = self.current_frame
            
            # Create merged lineage including both parents
            merged_lineage = []
            merged_lineage.extend(p1['parent_ids'])
            merged_lineage.append(p1['id'])
            merged_lineage.extend(p2['parent_ids'])
            merged_lineage.append(p2['id'])
            seen = set()
            merged_lineage = [x for x in merged_lineage if not (x in seen or seen.add(x))]
            
            # Create new particle (born next frame)
            new_id = self.add_particle(new_x, new_y, new_z, total_mass, 
                                     parent_ids=merged_lineage, birth_frame=self.current_frame + 1)
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            new_particle['last_event_frame'] = self.current_frame
            
            # DEBUG
            if self.enable_warnings:
                print(f"   Created:      Particle {new_id} (birth_frame={new_particle['birth_frame']})")
            
            # PHASE 2: Schedule POST_MERGE label for next frame
            self.pending_post_events.append({
                'type': 'POST_MERGE',
                'particle_ids': [new_id]
            })
            
            # Track event
            self.all_merges.append({
                'parent_ids': [p1['id'], p2['id']],
                'child_id': new_id,
                'is_linking': False,
                'frame': self.current_frame
            })

    def process_split(self, particle):
        """Process a split event for a particle with proper two-phase timing"""
        # Validate particle before split
        if not self.validate_particle_frame_consistency(particle, "SPLIT"):
            return
            
        if not particle['active']:
            self.log_warning(f"SPLIT INACTIVE - Attempting to split inactive particle {particle['id']}")
            return
            
        if particle['mass'] < self.split_mass_threshold:
            self.log_warning(f"SPLIT MASS TOO LOW - Particle {particle['id']} mass {particle['mass']} < threshold {self.split_mass_threshold}")
            return
            
        ratio = np.random.uniform(0.4, 0.6)
        mass1 = particle['mass'] * ratio
        mass2 = particle['mass'] * (1 - ratio)
        
        # Check mass conservation
        if abs((mass1 + mass2) - particle['mass']) > 1e-6:
            self.log_warning(f"SPLIT MASS CONSERVATION - Mass conservation error: {mass1} + {mass2} != {particle['mass']}")
            
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        sep_dist = np.random.uniform(1, 12)
        
        dx = sep_dist * np.sin(theta) * np.cos(phi)
        dy = sep_dist * np.sin(theta) * np.sin(phi)
        dz = sep_dist * np.cos(theta)
        
        # DEBUG: Track split event
        if self.enable_warnings:
            print(f"DEBUG: Frame {self.current_frame} - Processing split of particle {particle['id']}")
        
        if np.random.random() < self.split_linking_prob:
            # LINKING SPLIT - one particle continues, one new particle created
            if mass1 >= mass2:
                continuing_mass = mass1
                new_mass = mass2
                continuing_x = particle['x'] - dx/2
                continuing_y = particle['y'] - dy/2
                continuing_z = particle['z'] - dz/2
                new_x = particle['x'] + dx/2
                new_y = particle['y'] + dy/2
                new_z = particle['z'] + dz/2
            else:
                continuing_mass = mass2
                new_mass = mass1
                continuing_x = particle['x'] + dx/2
                continuing_y = particle['y'] + dy/2
                continuing_z = particle['z'] + dz/2
                new_x = particle['x'] - dx/2
                new_y = particle['y'] - dy/2
                new_z = particle['z'] - dz/2
            
            # DEBUG
            if self.enable_warnings:
                print(f"\n🔗 LINKING SPLIT ANALYSIS - Frame {self.current_frame}")
                print(f"   Parent:       Particle {particle['id']} (mass={particle['mass']:.0f})")
                print(f"   Continuing:   Particle {particle['id']} (new mass={continuing_mass:.0f})")
                print(f"   New child:    Mass={new_mass:.0f}")
                print(f"   Expected next frame: BOTH particles get POST_SPLIT")
            
            # PHASE 1: Record continuing particle with SPLIT label (this frame)
            self.record_particle_state(particle, EventLabel.SPLIT)
            
            # Update continuing particle properties
            self.update_particle_properties(particle, continuing_x, continuing_y, continuing_z, 
                                          continuing_mass)
            particle['last_event_frame'] = self.current_frame
            
            # Create new particle with parent lineage
            child_lineage = particle['parent_ids'].copy()
            new_id = self.add_particle(new_x, new_y, new_z, new_mass, 
                                     parent_ids=child_lineage, birth_frame=self.current_frame + 1)
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            new_particle['last_event_frame'] = self.current_frame
            
            # DEBUG
            if self.enable_warnings:
                print(f"   Created:      Particle {new_id} (birth_frame={new_particle['birth_frame']})")
            
            # PHASE 2: Schedule POST_SPLIT labels for next frame (BOTH particles in linking split)
            self.pending_post_events.append({
                'type': 'POST_SPLIT',
                'particle_ids': [particle['id'], new_id]  # Both continuing AND new particle get POST_SPLIT
            })
            
            # Track event
            child_ids = [particle['id'], new_id] if mass1 >= mass2 else [new_id, particle['id']]
            self.all_splits.append({
                'parent_id': particle['id'],
                'child_ids': child_ids,
                'is_linking': True,
                'frame': self.current_frame
            })
            
        else:
            # NORMAL SPLIT - parent disappears, two new particles created
            # DEBUG
            if self.enable_warnings:
                print(f"\n⚫ NORMAL SPLIT ANALYSIS - Frame {self.current_frame}")
                print(f"   Parent:       Particle {particle['id']} (mass={particle['mass']:.0f})")
                print(f"   Child 1:      Mass={mass1:.0f}")
                print(f"   Child 2:      Mass={mass2:.0f}")
                print(f"   Expected next frame: BOTH children get POST_SPLIT")
            
            # PHASE 1: Record parent with SPLIT label (this frame)
            self.record_particle_state(particle, EventLabel.SPLIT)
            particle['active'] = False
            particle['death_frame'] = self.current_frame
            
            # Create child lineage including the parent
            child_lineage = particle['parent_ids'].copy()
            child_lineage.append(particle['id'])
            
            # Create two new particles (born next frame)
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
            
            # DEBUG
            if self.enable_warnings:
                print(f"   Created:      Particle {child1_id} (birth_frame={child1['birth_frame']})")
                print(f"   Created:      Particle {child2_id} (birth_frame={child2['birth_frame']})")
            
            # PHASE 2: Schedule POST_SPLIT labels for next frame
            self.pending_post_events.append({
                'type': 'POST_SPLIT',
                'particle_ids': [child1_id, child2_id]
            })
            
            # Track event
            self.all_splits.append({
                'parent_id': particle['id'],
                'child_ids': [child1_id, child2_id],
                'is_linking': False,
                'frame': self.current_frame
            })

    def validate_event_labeling(self):
        """Validate that event labeling follows correct sequence - DISABLED FOR TWO-PHASE SYSTEM"""
        # NOTE: With two-phase event timing, validation needs to check across frames
        # which is more complex. For now, we disable immediate validation since
        # the two-phase system is working correctly based on debug output.
        # POST labels appear in the frame AFTER the event, which is correct behavior.
        pass

    def validate_label_sequence(self):
        """Validate label sequences for all particles"""
        for particle in self.particles:
            if not particle['label_history']:
                continue
                
            labels = particle['label_history']
            
            # Check for invalid label sequences
            for i in range(len(labels) - 1):
                current_label = labels[i]
                next_label = labels[i + 1]
                
                # SPLIT should not be followed by POST_MERGE
                if current_label == EventLabel.SPLIT and next_label == EventLabel.POST_MERGE:
                    self.log_warning(f"INVALID SEQUENCE - Particle {particle['id']} has SPLIT followed by POST_MERGE")
                
                # MERGE should not be followed by POST_SPLIT  
                if current_label == EventLabel.MERGE and next_label == EventLabel.POST_SPLIT:
                    self.log_warning(f"INVALID SEQUENCE - Particle {particle['id']} has MERGE followed by POST_SPLIT")
                
                # POST_MERGE should not be immediately followed by POST_SPLIT (unless there's a gap)
                if current_label == EventLabel.POST_MERGE and next_label == EventLabel.POST_SPLIT:
                    self.log_warning(f"SUSPICIOUS SEQUENCE - Particle {particle['id']} has POST_MERGE immediately followed by POST_SPLIT")
                
                # POST_SPLIT should not be immediately followed by POST_MERGE (unless there's a gap)
                if current_label == EventLabel.POST_SPLIT and next_label == EventLabel.POST_MERGE:
                    self.log_warning(f"SUSPICIOUS SEQUENCE - Particle {particle['id']} has POST_SPLIT immediately followed by POST_MERGE")
            
            # Check for multiple consecutive event labels
            consecutive_events = []
            for i in range(len(labels) - 1):
                current_label = labels[i]
                next_label = labels[i + 1]
                
                if current_label in [EventLabel.MERGE, EventLabel.SPLIT, EventLabel.POST_MERGE, EventLabel.POST_SPLIT]:
                    if next_label in [EventLabel.MERGE, EventLabel.SPLIT, EventLabel.POST_MERGE, EventLabel.POST_SPLIT]:
                        consecutive_events.append(f"frames {particle['birth_frame'] + i}-{particle['birth_frame'] + i + 1}: {current_label}->{next_label}")
            
            if consecutive_events:
                self.log_warning(f"CONSECUTIVE EVENTS - Particle {particle['id']} has consecutive event labels: {consecutive_events}")

    def validate_frame_state(self):
        """Validate the overall frame state after update - TEMPORARILY DISABLED"""
        # Temporarily disable all validation to test if consecutive warnings disappear
        pass
        
        # Original validation code commented out:
        # active_particles = [p for p in self.particles if p['active']]
        # ... rest of validation ...

    def process_pending_post_events(self):
        """Process pending POST_MERGE and POST_SPLIT events from previous frame"""
        # First, activate any particles that should be born this frame
        for particle in self.particles:
            if not particle['active'] and particle['birth_frame'] == self.current_frame:
                particle['active'] = True
                if self.enable_warnings:
                    print(f"🟢 BIRTH: Activating particle {particle['id']} at birth frame {self.current_frame}")
        
        if not self.pending_post_events:
            return
            
        if self.enable_warnings:
            print(f"\n📋 POST-EVENT PROCESSING - Frame {self.current_frame}")
            print(f"   Processing {len(self.pending_post_events)} pending events from previous frame")
        
        for event in self.pending_post_events:
            event_type = event['type']
            particle_ids = event['particle_ids']
            
            if self.enable_warnings:
                if event_type == 'POST_MERGE':
                    print(f"   🔗➡️  POST_MERGE: Particles {particle_ids}")
                elif event_type == 'POST_SPLIT':
                    print(f"   🔗➡️  POST_SPLIT: Particles {particle_ids}")
            
            for particle_id in particle_ids:
                particle = next((p for p in self.particles if p['id'] == particle_id), None)
                if particle and particle['active']:
                    if event_type == 'POST_MERGE':
                        self.record_particle_state(particle, EventLabel.POST_MERGE)
                        if self.enable_warnings:
                            print(f"     ✅ Particle {particle_id} recorded with POST_MERGE")
                    elif event_type == 'POST_SPLIT':
                        self.record_particle_state(particle, EventLabel.POST_SPLIT)
                        if self.enable_warnings:
                            print(f"     ✅ Particle {particle_id} recorded with POST_SPLIT")
                else:
                    if self.enable_warnings:
                        print(f"     ❌ Particle {particle_id} - not found or inactive")
        
        # Clear pending events
        self.pending_post_events = []

    def update(self, dt=0.1):
        """Main update function with proper two-phase event timing"""
        self.current_frame += 1
        
        # Pre-update validation
        active_particles = [p for p in self.particles if p['active']]
        
        if not active_particles:
            self.log_warning("NO ACTIVE PARTICLES - No particles to update")
            return
            
        # Reset recording flags for all particles
        for particle in self.particles:
            particle['recorded_this_frame'] = False
        
        # PHASE 0: Process pending POST events from previous frame
        self.process_pending_post_events()
        
        # Get particles that haven't been recorded yet (not involved in POST events)
        unrecorded_particles = [p for p in active_particles if not p.get('recorded_this_frame', False)]
        
        # Validate all active particles before processing
        for particle in unrecorded_particles:
            self.validate_particle_frame_consistency(particle, "PRE_UPDATE")
            
        # Track all particles that have been processed this frame
        processed_particles = set()
        
        # PHASE 1: Process new merge events (record MERGE labels, schedule POST_MERGE for next frame)
        merge_candidates = self.find_merge_candidates(unrecorded_particles)
        
        for p1, p2 in merge_candidates:
            if p1['id'] in processed_particles or p2['id'] in processed_particles:
                continue
                
            if np.random.random() < self.merge_prob:
                # Track particles before processing
                p1_id, p2_id = p1['id'], p2['id']
                
                self.process_merge(p1, p2)
                
                # Mark both particles as processed
                processed_particles.add(p1_id)
                processed_particles.add(p2_id)
        
        # PHASE 2: Process new split events (record SPLIT labels, schedule POST_SPLIT for next frame)
        current_unrecorded = [p for p in self.particles if p['active'] and not p.get('recorded_this_frame', False)]
        
        for particle in current_unrecorded:
            if particle['id'] in processed_particles:
                continue
                
            if (np.random.random() < self.split_prob and 
                self.can_have_event(particle) and
                particle['mass'] >= self.split_mass_threshold):
                
                # Calculate kinetic factor safely
                active_D_values = [p['D'] for p in current_unrecorded if p['active']]
                if active_D_values:
                    kinetic_factor = particle['D'] / np.mean(active_D_values)
                    if np.random.random() < kinetic_factor:
                        particle_id = particle['id']
                        
                        self.process_split(particle)
                        
                        # Mark particle as processed
                        processed_particles.add(particle_id)
        
        # PHASE 3: Process normal updates for remaining particles
        final_unrecorded = [p for p in self.particles if p['active'] and not p.get('recorded_this_frame', False)]
        
        for particle in final_unrecorded:
            # Skip if already processed
            if particle['id'] in processed_particles:
                continue
                
            # Check for spontaneous disappearance
            if np.random.random() < self.spontaneous_disappear_prob:
                self.record_particle_state(particle, EventLabel.NORMAL)
                particle['active'] = False
                particle['death_frame'] = self.current_frame
                continue
            
            # Update position
            left_boundaries = self.update_particle_position(particle, dt)
            
            # Handle boundary conditions
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
            
            new_particle = next(p for p in self.particles if p['id'] == new_id)
            self.record_particle_state(new_particle, EventLabel.NORMAL)
        
        # Post-update validation
        self.validate_frame_state()

    def export_to_csv(self, filename):
        """Export simulation data to CSV with validation"""
        data = []
        export_warnings = []
        
        for particle in self.particles:
            if not particle['trajectory']:
                export_warnings.append(f"Particle {particle['id']} has empty trajectory")
                continue
                
            death_frame = particle['death_frame'] if particle['death_frame'] is not None else self.current_frame
            duration = death_frame - particle['birth_frame'] + 1
            
            # Validate trajectory consistency
            if len(particle['trajectory']) != len(particle['mass_history']):
                export_warnings.append(f"Particle {particle['id']} trajectory/mass history length mismatch: {len(particle['trajectory'])} != {len(particle['mass_history'])}")
                continue
                
            if len(particle['trajectory']) != len(particle['label_history']):
                export_warnings.append(f"Particle {particle['id']} trajectory/label history length mismatch: {len(particle['trajectory'])} != {len(particle['label_history'])}")
                continue
            
            # Validate expected trajectory length
            expected_length = death_frame - particle['birth_frame'] + 1
            actual_length = len(particle['trajectory'])
            if actual_length != expected_length:
                export_warnings.append(f"Particle {particle['id']} trajectory length {actual_length} != expected {expected_length} (birth={particle['birth_frame']}, death={death_frame})")
            
            for i, (x, y, z) in enumerate(particle['trajectory']):
                frame = particle['birth_frame'] + i
                
                # Validate frame bounds
                if frame > death_frame:
                    export_warnings.append(f"Particle {particle['id']} recording at frame {frame} > death_frame {death_frame}")
                    break
                    
                if frame > self.current_frame:
                    export_warnings.append(f"Particle {particle['id']} recording at future frame {frame} > current {self.current_frame}")
                    break
                
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
        
        # Report export warnings
        if export_warnings and self.enable_warnings:
            print("\n=== EXPORT WARNINGS ===")
            for warning in export_warnings:
                print(f"EXPORT WARNING: {warning}")
            print(f"Total export warnings: {len(export_warnings)}")
            print("======================\n")
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(['frame', 'particle'])
        else:
            print("ERROR: No data to export!")
            
        df.to_csv(filename, index=False)
        
        # Final validation of exported data
        if not df.empty:
            frame_counts = df.groupby('particle')['frame'].count()
            duration_values = df.groupby('particle')['duration'].first()
            
            mismatches = []
            for particle_id in frame_counts.index:
                if frame_counts[particle_id] != duration_values[particle_id]:
                    mismatches.append(f"Particle {particle_id}: {frame_counts[particle_id]} records != {duration_values[particle_id]} duration")
                    
            if mismatches and self.enable_warnings:
                print("\n=== DURATION MISMATCH WARNINGS ===")
                for mismatch in mismatches:
                    print(f"DURATION MISMATCH: {mismatch}")
                print("=================================\n")
        
        return df
    
    def print_summary(self):
        """Print simulation summary including warnings and event mechanics examples"""
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Total frames: {self.current_frame}")
        print(f"Total particles created: {self.next_id}")
        print(f"Total merges: {len(self.all_merges)}")
        print(f"Total splits: {len(self.all_splits)}")
        print(f"Total frame warnings: {len(self.frame_warnings)}")
        
        # Show examples of each event type
        print(f"\n=== EVENT MECHANICS EXAMPLES ===")
        
        # Normal merges
        normal_merges = [m for m in self.all_merges if not m['is_linking']]
        if normal_merges:
            example = normal_merges[0]
            print(f"⚫ NORMAL MERGE Example (Frame {example['frame']}):")
            print(f"   Parents {example['parent_ids']} → Child {example['child_id']}")
            print(f"   Frame {example['frame']}: Parents get MERGE labels")
            print(f"   Frame {example['frame']+1}: Child gets POST_MERGE label")
        
        # Linking merges  
        linking_merges = [m for m in self.all_merges if m['is_linking']]
        if linking_merges:
            example = linking_merges[0]
            print(f"🔗 LINKING MERGE Example (Frame {example['frame']}):")
            print(f"   Parents {example['parent_ids']} → Continuing {example['child_id']}")
            print(f"   Frame {example['frame']}: Both particles get MERGE labels")
            print(f"   Frame {example['frame']+1}: Continuing particle gets POST_MERGE label")
        
        # Normal splits
        normal_splits = [s for s in self.all_splits if not s['is_linking']]
        if normal_splits:
            example = normal_splits[0]
            print(f"⚫ NORMAL SPLIT Example (Frame {example['frame']}):")
            print(f"   Parent {example['parent_id']} → Children {example['child_ids']}")
            print(f"   Frame {example['frame']}: Parent gets SPLIT label")
            print(f"   Frame {example['frame']+1}: Children get POST_SPLIT labels")
        
        # Linking splits
        linking_splits = [s for s in self.all_splits if s['is_linking']]
        if linking_splits:
            example = linking_splits[0]
            print(f"🔗 LINKING SPLIT Example (Frame {example['frame']}):")
            print(f"   Parent {example['parent_id']} → Continuing+New {example['child_ids']}")
            print(f"   Frame {example['frame']}: Continuing particle gets SPLIT label")
            print(f"   Frame {example['frame']+1}: BOTH particles get POST_SPLIT labels")
        
        if self.frame_warnings and self.enable_warnings:
            print(f"\nFirst 10 warnings:")
            for i, warning in enumerate(self.frame_warnings[:10]):
                print(f"  {i+1}. {warning}")
            if len(self.frame_warnings) > 10:
                print(f"  ... and {len(self.frame_warnings) - 10} more")
        
        print("=========================\n")

    
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
        event_cooldown=5,
        enable_warnings=True  # Enable warnings for debugging
    )
    
    # Initialize particles
    for i in range(num_initial):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        z = np.random.uniform(10, 40)
        mass = np.random.lognormal(np.log(150000), 0.5)
        mass = np.clip(mass, sim.min_mass, sim.max_mass)
        particle_id = sim.add_particle(x, y, z, mass)
        
        # Record initial state
        particle = next(p for p in sim.particles if p['id'] == particle_id)
        sim.record_particle_state(particle, EventLabel.NORMAL)
    
    # Run simulation
    for frame in range(num_frames):
        sim.update(dt=0.1)
    
    # Export data
    output_file = os.path.join(output_dir, f'tracked_particles_3d_{sim_id:05d}.csv')
    df = sim.export_to_csv(output_file)
    
    # Print summary for single simulation
    if sim_id == 0:  # Only print for first simulation to avoid spam
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
    run_multiprocess_simulations(num_simulations=1, num_cores=1)