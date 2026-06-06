import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
import time

# =========== NUMBA SETUP ==========
try:
    from numba import jit, njit, prange
    import numba as nb
    NUMBA_AVAILABLE = True
    print("✅ Numba available in packing_engine - JIT compilation ENABLED")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not available - install with: pip install numba")

# =========== CLASS DEFINITIONS ==========
class Package:
    def __init__(self, id_: str, length: float, width: float, height: float, weight: float):
        self.id = id_
        self.original = (length, width, height)
        self.weight = weight
        self.volume = length * width * height
        self.orientations = self.generate_orientations()

    def generate_orientations(self):
        l, w, h = self.original
        return {
            1: (l, w, h),
            2: (l, h, w),
            3: (w, l, h),
            4: (w, h, l),
            5: (h, l, w),
            6: (h, w, l)
        }

class Container:
    def __init__(self, length: float, width: float, height: float, max_weight: float):
        self.length = length
        self.width = width
        self.height = height
        self.max_weight = max_weight
        self.volume = length * width * height
        self.cog_limit_x = length * 0.1
        self.cog_limit_y = width * 0.1
        self.cog_limit_z = height * 0.15

class ChromosomeCache:
    """Cache untuk menyimpan hasil evaluasi kromosom yang sudah pernah dievaluasi"""
    
    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _to_key(self, chromosome):
        return tuple((pid, rot) for pid, rot in chromosome)
    
    def get(self, chromosome):
        key = self._to_key(chromosome)
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key].copy()
        self.miss_count += 1
        return None
    
    def put(self, chromosome, fitness, metadata):
        key = self._to_key(chromosome)
        cached_meta = {
            'fitness': metadata['fitness'],
            'volume_utilization': metadata['volume_utilization'],
            'weight_utilization': metadata['weight_utilization'],
            'total_volume': metadata['total_volume'],
            'total_weight': metadata['total_weight'],
            'penalty_cog': metadata.get('penalty_cog', 0),
            'B1': metadata.get('B1', 0),
            'B2': metadata.get('B2', 0),
            'B3': metadata.get('B3', 0),
            'B4': metadata['B4'],
            'B5': metadata['B5'],
            'num_placed': metadata['num_placed'],
            'center_of_gravity': metadata.get('center_of_gravity', (0, 0, 0)),
            'cog_deviation': metadata.get('cog_deviation', (0, 0, 0)),
        }
        self.cache[key] = cached_meta
    
    def stats(self):
        total = self.hit_count + self.miss_count
        if total == 0:
            return "Cache: 0 hits, 0 misses (0%)"
        hit_rate = (self.hit_count / total) * 100
        return f"Cache: {self.hit_count} hits, {self.miss_count} misses (Hit rate: {hit_rate:.1f}%)"
    
    def clear(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def size(self):
        return len(self.cache)


# =========== NUMBA DATA INITIALIZATION ==========
_numba_package_dims = None
_numba_package_weights = None
_numba_package_id_to_idx = None
_numba_initialized = False


def initialize_numba_data(packages_dict):
    global _numba_package_dims, _numba_package_weights, _numba_package_id_to_idx, _numba_initialized
    
    if not NUMBA_AVAILABLE:
        return False
    
    package_ids = list(packages_dict.keys())
    _numba_package_id_to_idx = {pid: i for i, pid in enumerate(package_ids)}
    
    n_packages = len(package_ids)
    _numba_package_dims = np.zeros((n_packages, 18), dtype=np.float32)
    _numba_package_weights = np.zeros(n_packages, dtype=np.float32)
    
    for pid, pkg in packages_dict.items():
        idx = _numba_package_id_to_idx[pid]
        _numba_package_weights[idx] = pkg.weight
        for rot in range(1, 7):
            dims = pkg.orientations[rot]
            _numba_package_dims[idx, (rot-1)*3] = dims[0]
            _numba_package_dims[idx, (rot-1)*3 + 1] = dims[1]
            _numba_package_dims[idx, (rot-1)*3 + 2] = dims[2]
    
    _numba_initialized = True
    print(f"✅ Numba data initialized for {n_packages} packages")
    return True


# =========== OPTIMIZED BLF WITH Z-PRIORITY & EARLY EXIT ==========
@njit(cache=True, fastmath=True)
def bottom_left_fill_numba_optimized(chromosome_ids, chromosome_rots, 
                                      package_dims_array, package_weights_array,
                                      container_len, container_wid, container_hei,
                                      container_cog_limit_x, container_cog_limit_y, container_cog_limit_z,
                                      container_max_weight):
    L, W, H = int(container_len), int(container_wid), int(container_hei)
    L_int, W_int, H_int = L + 1, W + 1, H + 1
    
    grid = np.zeros((L_int, W_int, H_int), dtype=np.uint8)
    skyline = np.zeros((L_int, W_int), dtype=np.int16)
    support_map = np.zeros((L_int, W_int), dtype=np.int16)
    
    n_packages = len(chromosome_ids)
    
    pos_x = np.zeros(n_packages, dtype=np.float32)
    pos_y = np.zeros(n_packages, dtype=np.float32)
    pos_z = np.zeros(n_packages, dtype=np.float32)
    placed_flags = np.zeros(n_packages, dtype=np.int8)
    package_volumes = np.zeros(n_packages, dtype=np.float32)
    
    total_volume = 0.0
    total_weight = 0.0
    total_mass_x = 0.0
    total_mass_y = 0.0
    total_mass_z = 0.0
    all_stability_valid = True
    num_placed = 0
    
    for idx in range(n_packages):
        p_id = chromosome_ids[idx]
        rot = chromosome_rots[idx]
        
        base_idx = rot * 3
        dx = package_dims_array[p_id, base_idx]
        dy = package_dims_array[p_id, base_idx + 1]
        dz = package_dims_array[p_id, base_idx + 2]
        
        dx_int = int(np.ceil(dx))
        dy_int = int(np.ceil(dy))
        dz_int = int(np.ceil(dz))
        
        max_x = L - dx_int + 1
        max_y = W - dy_int + 1
        
        placed = False
        best_x, best_y, best_z = -1, -1, -1
        
        min_z_global = 999999
        best_y_candidate = 999999
        best_x_candidate = 999999
        found_zero_z = False
        
        for y in range(max_y):
            if found_zero_z:
                break
            
            for x in range(max_x):
                z_min = 0
                for i in range(dx_int):
                    for j in range(dy_int):
                        val = skyline[x + i, y + j]
                        if val > z_min:
                            z_min = val
                            if z_min >= min_z_global:
                                break
                    if z_min >= min_z_global:
                        break
                
                if z_min >= min_z_global and min_z_global != 999999:
                    continue
                
                if z_min + dz_int > H:
                    continue
                
                collision = False
                for i in range(dx_int):
                    for j in range(dy_int):
                        for k in range(dz_int):
                            if grid[x + i, y + j, z_min + k] != 0:
                                collision = True
                                break
                        if collision:
                            break
                    if collision:
                        break
                
                if collision:
                    max_z_in_area = 0
                    for i in range(dx_int):
                        for j in range(dy_int):
                            for k in range(dz_int):
                                if grid[x + i, y + j, z_min + k] != 0:
                                    found_z = z_min + k + 1
                                    if found_z > max_z_in_area:
                                        max_z_in_area = found_z
                    z_candidate = max_z_in_area
                    
                    if z_candidate + dz_int > H:
                        continue
                    
                    collision = False
                    for i in range(dx_int):
                        for j in range(dy_int):
                            for k in range(dz_int):
                                if grid[x + i, y + j, z_candidate + k] != 0:
                                    collision = True
                                    break
                            if collision:
                                break
                        if collision:
                            break
                    
                    if collision:
                        continue
                else:
                    z_candidate = z_min
                
                stability_valid = True
                if z_candidate > 0:
                    support_count = 0
                    for i in range(dx_int):
                        for j in range(dy_int):
                            if support_map[x + i, y + j] > 0:
                                support_count += 1
                    
                    if support_count < (dx_int * dy_int) * 0.5:
                        stability_valid = False
                        all_stability_valid = False
                
                if not stability_valid:
                    continue
                
                if z_candidate < min_z_global:
                    min_z_global = z_candidate
                    best_y_candidate = y
                    best_x_candidate = x
                    
                    if min_z_global == 0:
                        found_zero_z = True
                        break
                
                elif z_candidate == min_z_global:
                    if y < best_y_candidate:
                        best_y_candidate = y
                        best_x_candidate = x
                    elif y == best_y_candidate and x < best_x_candidate:
                        best_x_candidate = x
        
        if min_z_global != 999999:
            best_x = best_x_candidate
            best_y = best_y_candidate
            best_z = min_z_global
            placed = True
            
            for i in range(dx_int):
                for j in range(dy_int):
                    for k in range(dz_int):
                        grid[best_x + i, best_y + j, best_z + k] = p_id + 1
            
            new_top = best_z + dz_int
            for i in range(dx_int):
                for j in range(dy_int):
                    if new_top > skyline[best_x + i, best_y + j]:
                        skyline[best_x + i, best_y + j] = new_top
            
            for i in range(dx_int):
                for j in range(dy_int):
                    support_map[best_x + i, best_y + j] += 1
            
            volume = dx * dy * dz
            weight = package_weights_array[p_id]
            cog_x = best_x + dx / 2.0
            cog_y = best_y + dy / 2.0
            cog_z = best_z + dz / 2.0
            
            total_volume += volume
            total_weight += weight
            total_mass_x += weight * cog_x
            total_mass_y += weight * cog_y
            total_mass_z += weight * cog_z
            num_placed += 1
            
            pos_x[idx] = best_x
            pos_y[idx] = best_y
            pos_z[idx] = best_z
            placed_flags[idx] = 1
            package_volumes[idx] = volume
        
        if not placed:
            placed_flags[idx] = 0
            package_volumes[idx] = dx * dy * dz
            pos_x[idx] = -1
            pos_y[idx] = -1
            pos_z[idx] = -1
    
    if total_weight > 0:
        cog_total_x = total_mass_x / total_weight
        cog_total_y = total_mass_y / total_weight
        cog_total_z = total_mass_z / total_weight
    else:
        cog_total_x = container_len / 2.0
        cog_total_y = container_wid / 2.0
        cog_total_z = container_hei / 2.0
    
    container_center_x = container_len / 2.0
    container_center_y = container_wid / 2.0
    container_center_z = container_hei / 2.0
    
    dev_x = abs(cog_total_x - container_center_x)
    dev_y = abs(cog_total_y - container_center_y)
    dev_z = abs(cog_total_z - container_center_z)
    
    B1 = max(0.0, dev_x - container_cog_limit_x)
    B2 = max(0.0, dev_y - container_cog_limit_y)
    B3 = max(0.0, dev_z - container_cog_limit_z)
    
    penalty_cog = B1 + B2 + B3
    B4 = 1 if total_weight <= container_max_weight else 0
    B5 = 1 if all_stability_valid else 0
    
    fitness_raw = total_volume - penalty_cog
    fitness_final = fitness_raw * B4 * B5
    
    return (fitness_final, total_volume, total_weight, penalty_cog, 
            B1, B2, B3, B4, B5, num_placed,
            cog_total_x, cog_total_y, cog_total_z, dev_x, dev_y, dev_z,
            pos_x, pos_y, pos_z, placed_flags, package_volumes)


# =========== PARALLEL BATCH EVALUATION ==========
@njit(parallel=True, cache=True, fastmath=True)
def evaluate_population_parallel_numba(chromosomes_ids_batch, chromosomes_rots_batch,
                                         package_dims_array, package_weights_array,
                                         container_len, container_wid, container_hei,
                                         container_cog_limit_x, container_cog_limit_y, container_cog_limit_z,
                                         container_max_weight):
    n_chrom = chromosomes_ids_batch.shape[0]
    n_packages = chromosomes_ids_batch.shape[1]
    
    all_fitness = np.zeros(n_chrom, dtype=np.float64)
    all_total_volume = np.zeros(n_chrom, dtype=np.float64)
    all_total_weight = np.zeros(n_chrom, dtype=np.float64)
    all_penalty_cog = np.zeros(n_chrom, dtype=np.float64)
    all_B1 = np.zeros(n_chrom, dtype=np.float64)
    all_B2 = np.zeros(n_chrom, dtype=np.float64)
    all_B3 = np.zeros(n_chrom, dtype=np.float64)
    all_B4 = np.zeros(n_chrom, dtype=np.int8)
    all_B5 = np.zeros(n_chrom, dtype=np.int8)
    all_num_placed = np.zeros(n_chrom, dtype=np.int32)
    all_cog_x = np.zeros(n_chrom, dtype=np.float64)
    all_cog_y = np.zeros(n_chrom, dtype=np.float64)
    all_cog_z = np.zeros(n_chrom, dtype=np.float64)
    
    for c in prange(n_chrom):
        chrom_ids = chromosomes_ids_batch[c]
        chrom_rots = chromosomes_rots_batch[c]
        
        L, W, H = int(container_len), int(container_wid), int(container_hei)
        L_int, W_int, H_int = L + 1, W + 1, H + 1
        
        grid = np.zeros((L_int, W_int, H_int), dtype=np.uint8)
        skyline = np.zeros((L_int, W_int), dtype=np.int16)
        support_map = np.zeros((L_int, W_int), dtype=np.int16)
        
        total_volume = 0.0
        total_weight = 0.0
        total_mass_x = 0.0
        total_mass_y = 0.0
        total_mass_z = 0.0
        all_stability_valid = True
        num_placed = 0
        
        for idx in range(n_packages):
            p_id = chrom_ids[idx]
            rot = chrom_rots[idx]
            
            base_idx = rot * 3
            dx = package_dims_array[p_id, base_idx]
            dy = package_dims_array[p_id, base_idx + 1]
            dz = package_dims_array[p_id, base_idx + 2]
            
            dx_int = int(np.ceil(dx))
            dy_int = int(np.ceil(dy))
            dz_int = int(np.ceil(dz))
            
            max_x = L - dx_int + 1
            max_y = W - dy_int + 1
            
            placed = False
            best_x, best_y, best_z = -1, -1, -1
            
            min_z_global = 999999
            best_y_candidate = 999999
            best_x_candidate = 999999
            found_zero_z = False
            
            for y in range(max_y):
                if found_zero_z:
                    break
                
                for x in range(max_x):
                    z_min = 0
                    for i in range(dx_int):
                        for j in range(dy_int):
                            val = skyline[x + i, y + j]
                            if val > z_min:
                                z_min = val
                                if z_min >= min_z_global:
                                    break
                        if z_min >= min_z_global:
                            break
                    
                    if z_min >= min_z_global and min_z_global != 999999:
                        continue
                    
                    if z_min + dz_int > H:
                        continue
                    
                    collision = False
                    for i in range(dx_int):
                        for j in range(dy_int):
                            for k in range(dz_int):
                                if grid[x + i, y + j, z_min + k] != 0:
                                    collision = True
                                    break
                            if collision:
                                break
                        if collision:
                            break
                    
                    if collision:
                        max_z_in_area = 0
                        for i in range(dx_int):
                            for j in range(dy_int):
                                for k in range(dz_int):
                                    if grid[x + i, y + j, z_min + k] != 0:
                                        found_z = z_min + k + 1
                                        if found_z > max_z_in_area:
                                            max_z_in_area = found_z
                        z_candidate = max_z_in_area
                        
                        if z_candidate + dz_int > H:
                            continue
                        
                        collision = False
                        for i in range(dx_int):
                            for j in range(dy_int):
                                for k in range(dz_int):
                                    if grid[x + i, y + j, z_candidate + k] != 0:
                                        collision = True
                                        break
                                if collision:
                                    break
                            if collision:
                                break
                        
                        if collision:
                            continue
                    else:
                        z_candidate = z_min
                    
                    stability_valid = True
                    if z_candidate > 0:
                        support_count = 0
                        for i in range(dx_int):
                            for j in range(dy_int):
                                if support_map[x + i, y + j] > 0:
                                    support_count += 1
                        
                        if support_count < (dx_int * dy_int) * 0.5:
                            stability_valid = False
                            all_stability_valid = False
                    
                    if not stability_valid:
                        continue
                    
                    if z_candidate < min_z_global:
                        min_z_global = z_candidate
                        best_y_candidate = y
                        best_x_candidate = x
                        
                        if min_z_global == 0:
                            found_zero_z = True
                            break
                    
                    elif z_candidate == min_z_global:
                        if y < best_y_candidate:
                            best_y_candidate = y
                            best_x_candidate = x
                        elif y == best_y_candidate and x < best_x_candidate:
                            best_x_candidate = x
            
            if min_z_global != 999999:
                best_x = best_x_candidate
                best_y = best_y_candidate
                best_z = min_z_global
                placed = True
                
                for i in range(dx_int):
                    for j in range(dy_int):
                        for k in range(dz_int):
                            grid[best_x + i, best_y + j, best_z + k] = p_id + 1
                
                new_top = best_z + dz_int
                for i in range(dx_int):
                    for j in range(dy_int):
                        if new_top > skyline[best_x + i, best_y + j]:
                            skyline[best_x + i, best_y + j] = new_top
                
                for i in range(dx_int):
                    for j in range(dy_int):
                        support_map[best_x + i, best_y + j] += 1
                
                volume = dx * dy * dz
                weight = package_weights_array[p_id]
                cog_x = best_x + dx / 2.0
                cog_y = best_y + dy / 2.0
                cog_z = best_z + dz / 2.0
                
                total_volume += volume
                total_weight += weight
                total_mass_x += weight * cog_x
                total_mass_y += weight * cog_y
                total_mass_z += weight * cog_z
                num_placed += 1
        
        if total_weight > 0:
            cog_total_x = total_mass_x / total_weight
            cog_total_y = total_mass_y / total_weight
            cog_total_z = total_mass_z / total_weight
        else:
            cog_total_x = container_len / 2.0
            cog_total_y = container_wid / 2.0
            cog_total_z = container_hei / 2.0
        
        container_center_x = container_len / 2.0
        container_center_y = container_wid / 2.0
        container_center_z = container_hei / 2.0
        
        dev_x = abs(cog_total_x - container_center_x)
        dev_y = abs(cog_total_y - container_center_y)
        dev_z = abs(cog_total_z - container_center_z)
        
        B1 = max(0.0, dev_x - container_cog_limit_x)
        B2 = max(0.0, dev_y - container_cog_limit_y)
        B3 = max(0.0, dev_z - container_cog_limit_z)
        
        penalty_cog = B1 + B2 + B3
        B4 = 1 if total_weight <= container_max_weight else 0
        B5 = 1 if all_stability_valid else 0
        
        fitness_raw = total_volume - penalty_cog
        fitness_final = fitness_raw * B4 * B5
        
        all_fitness[c] = fitness_final
        all_total_volume[c] = total_volume
        all_total_weight[c] = total_weight
        all_penalty_cog[c] = penalty_cog
        all_B1[c] = B1
        all_B2[c] = B2
        all_B3[c] = B3
        all_B4[c] = B4
        all_B5[c] = B5
        all_num_placed[c] = num_placed
        all_cog_x[c] = cog_total_x
        all_cog_y[c] = cog_total_y
        all_cog_z[c] = cog_total_z
    
    return (all_fitness, all_total_volume, all_total_weight, all_penalty_cog,
            all_B1, all_B2, all_B3, all_B4, all_B5, all_num_placed,
            all_cog_x, all_cog_y, all_cog_z)


# =========== WRAPPER EVALUASI POPULASI ==========
def evaluate_population_with_cache(population, container, packages_dict, cache):
    global _numba_initialized
    
    if not NUMBA_AVAILABLE:
        fitness_scores = []
        metadata_list = []
        for chrom in population:
            cached = cache.get(chrom)
            if cached:
                fitness_scores.append(cached['fitness'])
                metadata_list.append(cached)
            else:
                result = bottom_left_fill_with_fitness_fallback(chrom, container, packages_dict)
                fitness_scores.append(result['fitness'])
                metadata_list.append(result)
                cache.put(chrom, result['fitness'], result)
        return fitness_scores, metadata_list
    
    if not _numba_initialized:
        initialize_numba_data(packages_dict)
    
    n_chrom = len(population)
    if n_chrom == 0:
        return [], []
    
    fitness_scores = [0.0] * n_chrom
    metadata_list = [None] * n_chrom
    
    chromosomes_to_eval = []
    indices_to_eval = []
    
    for i, chrom in enumerate(population):
        cached = cache.get(chrom)
        if cached:
            fitness_scores[i] = cached['fitness']
            metadata_list[i] = cached
        else:
            chromosomes_to_eval.append(chrom)
            indices_to_eval.append(i)
    
    if chromosomes_to_eval:
        n_new = len(chromosomes_to_eval)
        n_packages = len(chromosomes_to_eval[0])
        
        chrom_ids_batch = np.zeros((n_new, n_packages), dtype=np.int32)
        chrom_rots_batch = np.zeros((n_new, n_packages), dtype=np.int32)
        
        for i, chrom in enumerate(chromosomes_to_eval):
            for j, (pid, rot) in enumerate(chrom):
                chrom_ids_batch[i, j] = _numba_package_id_to_idx[pid]
                chrom_rots_batch[i, j] = rot - 1
        
        (all_fitness, all_total_volume, all_total_weight, all_penalty_cog,
         all_B1, all_B2, all_B3, all_B4, all_B5, all_num_placed,
         all_cog_x, all_cog_y, all_cog_z) = evaluate_population_parallel_numba(
            chrom_ids_batch, chrom_rots_batch,
            _numba_package_dims, _numba_package_weights,
            container.length, container.width, container.height,
            container.cog_limit_x, container.cog_limit_y, container.cog_limit_z,
            container.max_weight
        )
        
        container_volume = container.volume
        
        for idx_in_batch, (chrom, original_idx) in enumerate(zip(chromosomes_to_eval, indices_to_eval)):
            fitness = float(all_fitness[idx_in_batch])
            total_volume = float(all_total_volume[idx_in_batch])
            total_weight = float(all_total_weight[idx_in_batch])
            penalty_cog = float(all_penalty_cog[idx_in_batch])
            B1 = float(all_B1[idx_in_batch])
            B2 = float(all_B2[idx_in_batch])
            B3 = float(all_B3[idx_in_batch])
            B4 = int(all_B4[idx_in_batch])
            B5 = int(all_B5[idx_in_batch])
            num_placed = int(all_num_placed[idx_in_batch])
            cog_x = float(all_cog_x[idx_in_batch])
            cog_y = float(all_cog_y[idx_in_batch])
            cog_z = float(all_cog_z[idx_in_batch])
            
            volume_utilization = (total_volume / container_volume) * 100 if container_volume > 0 else 0
            weight_utilization = (total_weight / container.max_weight) * 100 if container.max_weight > 0 else 0
            
            positions = []
            for j, (pid, rot) in enumerate(chrom):
                dims = packages_dict[pid].orientations[rot]
                positions.append({
                    'id': pid,
                    'x': -1, 'y': -1, 'z': -1,
                    'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
                    'weight': packages_dict[pid].weight,
                    'volume': dims[0] * dims[1] * dims[2],
                    'orientation': rot,
                    'placed': j < num_placed
                })
            
            metadata = {
                'fitness': fitness,
                'volume_utilization': volume_utilization,
                'weight_utilization': weight_utilization,
                'total_volume': total_volume,
                'total_weight': total_weight,
                'penalty_cog': penalty_cog,
                'B1': B1, 'B2': B2, 'B3': B3,
                'B4': B4, 'B5': B5,
                'num_placed': num_placed,
                'positions': positions,
                'center_of_gravity': (cog_x, cog_y, cog_z),
                'cog_deviation': (abs(cog_x - container.length/2), 
                                  abs(cog_y - container.width/2), 
                                  abs(cog_z - container.height/2))
            }
            
            cache.put(chrom, fitness, metadata)
            
            fitness_scores[original_idx] = fitness
            metadata_list[original_idx] = metadata
    
    return fitness_scores, metadata_list


# =========== FALLBACK BLF ==========
def bottom_left_fill_with_fitness_fallback(chromosome, container, packages_dict):
    L, W, H = int(container.length), int(container.width), int(container.height)
    
    positions = []
    placed = []
    space_grid = np.zeros((L, W, H), dtype=int)
    
    total_mass = 0
    total_mass_x = 0
    total_mass_y = 0
    total_mass_z = 0
    total_volume = 0
    all_stability_valid = True
    total_weight_placed = 0
    
    for gene in chromosome:
        p_id = gene[0]
        orientation = gene[1]
        package = packages_dict[p_id]
        dims = package.orientations[orientation]
        placed_flag = False
        
        dx_int, dy_int, dz_int = int(dims[0]), int(dims[1]), int(dims[2])
        
        for z in range(H - dz_int + 1):
            for y in range(W - dy_int + 1):
                for x in range(L - dx_int + 1):
                    if np.all(space_grid[x:x+dx_int, y:y+dy_int, z:z+dz_int] == 0):
                        stability_valid = True
                        if z > 0:
                            support_area = 0
                            total_area = dx_int * dy_int
                            for xp in range(x, x + dx_int):
                                for yp in range(y, y + dy_int):
                                    if space_grid[xp, yp, z-1] != 0:
                                        support_area += 1
                            if support_area < 0.5 * total_area:
                                stability_valid = False
                                all_stability_valid = False
                        
                        if not stability_valid:
                            continue
                        
                        space_grid[x:x+dx_int, y:y+dy_int, z:z+dz_int] = 1
                        
                        cog_x = x + dims[0] / 2.0
                        cog_y = y + dims[1] / 2.0
                        cog_z = z + dims[2] / 2.0
                        
                        total_mass += package.weight
                        total_mass_x += package.weight * cog_x
                        total_mass_y += package.weight * cog_y
                        total_mass_z += package.weight * cog_z
                        
                        volume = dims[0] * dims[1] * dims[2]
                        total_volume += volume
                        total_weight_placed += package.weight
                        
                        positions.append({
                            'id': p_id,
                            'x': x, 'y': y, 'z': z,
                            'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
                            'weight': package.weight,
                            'volume': volume,
                            'orientation': orientation,
                            'placed': True
                        })
                        placed.append(p_id)
                        placed_flag = True
                        break
                if placed_flag:
                    break
            if placed_flag:
                break
        
        if not placed_flag:
            dims = package.orientations[orientation]
            positions.append({
                'id': p_id,
                'x': -1, 'y': -1, 'z': -1,
                'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
                'weight': package.weight,
                'volume': dims[0] * dims[1] * dims[2],
                'orientation': orientation,
                'placed': False
            })
    
    B4 = 1 if total_weight_placed <= container.max_weight else 0
    B5 = 1 if all_stability_valid else 0
    
    if total_mass > 0:
        cog_total_x = total_mass_x / total_mass
        cog_total_y = total_mass_y / total_mass
        cog_total_z = total_mass_z / total_mass
    else:
        cog_total_x = cog_total_y = cog_total_z = 0
    
    container_center_x = container.length / 2.0
    container_center_y = container.width / 2.0
    container_center_z = container.height / 2.0
    
    dev_x = abs(cog_total_x - container_center_x)
    dev_y = abs(cog_total_y - container_center_y)
    dev_z = abs(cog_total_z - container_center_z)
    
    B1 = max(0, dev_x - container.cog_limit_x)
    B2 = max(0, dev_y - container.cog_limit_y)
    B3 = max(0, dev_z - container.cog_limit_z)
    
    penalty_cog = B1 + B2 + B3
    fitness_raw = total_volume - penalty_cog
    fitness_final = fitness_raw * B4 * B5
    
    volume_utilization = (total_volume / container.volume) * 100 if container.volume > 0 else 0
    weight_utilization = (total_weight_placed / container.max_weight) * 100 if container.max_weight > 0 else 0
    
    return {
        'fitness': float(fitness_final),
        'volume_utilization': float(volume_utilization),
        'weight_utilization': float(weight_utilization),
        'total_volume': float(total_volume),
        'total_weight': float(total_weight_placed),
        'num_placed': len(placed),
        'positions': positions,
        'center_of_gravity': [float(cog_total_x), float(cog_total_y), float(cog_total_z)],
        'B4': int(B4),
        'B5': int(B5)
    }


# =========== FUNGSI GENETIC ALGORITHM ==========
def create_chromosome(packages_list):
    ids = [p.id for p in packages_list]
    random.shuffle(ids)
    chromosome = [(p_id, random.randint(1, 6)) for p_id in ids]
    return chromosome


def tournament_selection(population, fitness_scores, tournament_size=3):
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitness_scores[i])
    return best_idx


def pmx_crossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return parent1.copy()
    
    p1 = parent1.copy()
    p2 = parent2.copy()
    
    cut1 = random.randint(0, size - 2)
    cut2 = random.randint(cut1 + 1, size - 1)
    
    child = [None] * size
    child[cut1:cut2] = p1[cut1:cut2]
    
    mapping = {}
    for i in range(cut1, cut2):
        mapping[p1[i][0]] = p2[i][0]
    
    for i in range(size):
        if i < cut1 or i >= cut2:
            gene = p2[i]
            used_ids = {g[0] for g in child if g is not None}
            while gene[0] in used_ids:
                if gene[0] in mapping:
                    mapped_id = mapping[gene[0]]
                    for g in p2:
                        if g[0] == mapped_id:
                            gene = g
                            break
                else:
                    for g in p1:
                        if g[0] not in used_ids:
                            gene = g
                            break
            child[i] = gene
    
    return child


def mutate(chromosome):
    mutated = chromosome.copy()
    if len(mutated) == 0:
        return mutated
    
    mutation_type = random.choice(['swap_order', 'swap_rotation'])
    
    if mutation_type == 'swap_order':
        if len(mutated) >= 2:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    else:
        idx = random.randint(0, len(mutated)-1)
        old_orient = mutated[idx][1]
        new_orient = random.randint(1, 6)
        while new_orient == old_orient:
            new_orient = random.randint(1, 6)
        mutated[idx] = (mutated[idx][0], new_orient)
    
    return mutated


def get_positions_from_best_chromosome(best_chromosome, container, packages_dict):
    if NUMBA_AVAILABLE and _numba_initialized:
        n = len(best_chromosome)
        chrom_ids = np.zeros(n, dtype=np.int32)
        chrom_rots = np.zeros(n, dtype=np.int32)
        
        for i, (pid, rot) in enumerate(best_chromosome):
            chrom_ids[i] = _numba_package_id_to_idx[pid]
            chrom_rots[i] = rot - 1
        
        (fitness, total_volume, total_weight, penalty_cog,
         B1, B2, B3, B4, B5, num_placed,
         cog_total_x, cog_total_y, cog_total_z, dev_x, dev_y, dev_z,
         pos_x, pos_y, pos_z, placed_flags, package_volumes) = bottom_left_fill_numba_optimized(
            chrom_ids, chrom_rots,
            _numba_package_dims, _numba_package_weights,
            container.length, container.width, container.height,
            container.cog_limit_x, container.cog_limit_y, container.cog_limit_z,
            container.max_weight
        )
        
        positions = []
        for i, (pid, rot) in enumerate(best_chromosome):
            dims = packages_dict[pid].orientations[rot]
            if placed_flags[i] == 1:
                positions.append({
                    'id': pid,
                    'x': float(pos_x[i]), 'y': float(pos_y[i]), 'z': float(pos_z[i]),
                    'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
                    'weight': packages_dict[pid].weight,
                    'volume': float(package_volumes[i]),
                    'orientation': rot,
                    'placed': True
                })
            else:
                positions.append({
                    'id': pid,
                    'x': -1, 'y': -1, 'z': -1,
                    'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
                    'weight': packages_dict[pid].weight,
                    'volume': dims[0] * dims[1] * dims[2],
                    'orientation': rot,
                    'placed': False
                })
        
        return positions, cog_total_x, cog_total_y, cog_total_z, total_volume, total_weight, num_placed, fitness
    
    else:
        result = bottom_left_fill_with_fitness_fallback(best_chromosome, container, packages_dict)
        positions = result['positions']
        cog_total_x, cog_total_y, cog_total_z = result['center_of_gravity']
        total_volume = result['total_volume']
        total_weight = result['total_weight']
        num_placed = result['num_placed']
        fitness = result['fitness']
        return positions, cog_total_x, cog_total_y, cog_total_z, total_volume, total_weight, num_placed, fitness


def run_genetic_algorithm(packages_data, container_data, params):
    packages = [Package(p['id'], p['length'], p['width'], p['height'], p['weight']) for p in packages_data]
    container = Container(
        container_data['length'], 
        container_data['width'], 
        container_data['height'], 
        container_data['max_weight']
    )
    packages_dict = {p.id: p for p in packages}
    
    population_size = params.get('population_size', 50)
    generations = params.get('generations', 50)
    crossover_rate = params.get('crossover_rate', 0.8)
    mutation_rate = params.get('mutation_rate', 0.2)
    
    print(f"\n🚀 Starting GA with Cache:")
    print(f"   Population: {population_size}, Generations: {generations}")
    print(f"   Crossover: {crossover_rate}, Mutation: {mutation_rate}")
    
    if NUMBA_AVAILABLE:
        initialize_numba_data(packages_dict)
    
    population = [create_chromosome(packages) for _ in range(population_size)]
    cache = ChromosomeCache()
    
    best_solution = None
    best_fitness = -float('inf')
    
    # === TAMBAH: Array untuk menyimpan history per generasi ===
    history = []
    
    start_time = time.time()
    
    for gen in range(generations):
        fitness_scores, metadata_list = evaluate_population_with_cache(population, container, packages_dict, cache)
        
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_best_metadata = metadata_list[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_metadata.copy()
            best_solution['chromosome'] = population[gen_best_idx].copy()
        
        # === TAMBAH: Simpan history untuk generasi ini ===
        history.append({
            'generation': gen + 1,
            'best_fitness': float(gen_best_fitness),
            'avg_fitness': float(np.mean(fitness_scores)),
            'best_volume_utilization': float(gen_best_metadata['volume_utilization']),
            'best_num_placed': gen_best_metadata['num_placed']
        })
        
        if (gen + 1) % 10 == 0 or gen == 0 or gen == generations - 1:
            avg_fitness = np.mean(fitness_scores)
            print(f"   Gen {gen+1:3d}: Best={best_fitness:8.2f}, Avg={avg_fitness:8.2f}, "
                  f"Placed={best_solution['num_placed']:2d}/{len(packages)}, "
                  f"Cache: {cache.size()} unique, {cache.stats()}")
        
        if gen < generations - 1:
            target_crossover_children = int(population_size * crossover_rate)
            target_mutation_children = population_size - target_crossover_children
            
            offspring = []
            
            for _ in range(target_crossover_children):
                p1_idx = tournament_selection(population, fitness_scores)
                p2_idx = tournament_selection(population, fitness_scores)
                while p2_idx == p1_idx and len(population) > 1:
                    p2_idx = tournament_selection(population, fitness_scores)
                child = pmx_crossover(population[p1_idx], population[p2_idx])
                offspring.append(child)
            
            for _ in range(target_mutation_children):
                p_idx = tournament_selection(population, fitness_scores)
                child = mutate(population[p_idx])
                offspring.append(child)
            
            combined_population = population + offspring
            combined_fitness = fitness_scores + [metadata_list[i]['fitness'] for i in range(len(offspring))]
            
            sorted_indices = np.argsort(combined_fitness)[::-1]
            best_indices = sorted_indices[:population_size]
            population = [combined_population[i] for i in best_indices]
    
    elapsed_time = time.time() - start_time
    
    if best_solution and 'chromosome' in best_solution:
        positions, cog_x, cog_y, cog_z, total_volume, total_weight, num_placed, fitness = get_positions_from_best_chromosome(
            best_solution['chromosome'], container, packages_dict
        )
        
        volume_utilization = (total_volume / container.volume) * 100 if container.volume > 0 else 0
        weight_utilization = (total_weight / container.max_weight) * 100 if container.max_weight > 0 else 0
        
        best_solution['positions'] = positions
        best_solution['center_of_gravity'] = [float(cog_x), float(cog_y), float(cog_z)]
        best_solution['total_volume'] = float(total_volume)
        best_solution['total_weight'] = float(total_weight)
        best_solution['num_placed'] = num_placed
        best_solution['fitness'] = float(fitness)
        best_solution['volume_utilization'] = float(volume_utilization)
        best_solution['weight_utilization'] = float(weight_utilization)
    
    # === TAMBAH: Masukkan history ke result ===
    best_solution['history'] = history
    best_solution['chromosome'] = best_solution.get('chromosome', [])
    best_solution['execution_time_seconds'] = elapsed_time
    
    best_solution['cache_stats'] = {
        'unique_chromosomes': cache.size(),
        'hit_rate': (cache.hit_count / (cache.hit_count + cache.miss_count) * 100) if (cache.hit_count + cache.miss_count) > 0 else 0,
        'total_hits': cache.hit_count,
        'total_misses': cache.miss_count
    }
    
    print(f"\n✅ GA Selesai! Waktu: {elapsed_time:.2f} detik")
    print(f"   Best Fitness: {best_fitness:.2f}")
    print(f"   Volume Utilization: {volume_utilization:.2f}%")
    print(f"   Cache Hit Rate: {best_solution['cache_stats']['hit_rate']:.1f}%")
    print(f"   Unique Chromosomes: {cache.size()}")
    
    return best_solution