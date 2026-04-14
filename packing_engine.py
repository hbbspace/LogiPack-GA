import numpy as np
import random
from typing import List, Dict, Any, Tuple

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

def bottom_left_fill_with_fitness(chromosome, container, packages_dict):
    positions = []
    placed = []
    space_grid = np.zeros((int(container.length), int(container.width), int(container.height)), dtype=int)

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

        for z in range(int(container.height - dims[2]) + 1):
            for y in range(int(container.width - dims[1]) + 1):
                for x in range(int(container.length - dims[0]) + 1):
                    if np.all(space_grid[x:x+int(dims[0]), y:y+int(dims[1]), z:z+int(dims[2])] == 0):
                        stability_valid = True
                        if z > 0:
                            support_area = 0
                            total_area = dims[0] * dims[1]
                            for xp in range(x, x + int(dims[0])):
                                for yp in range(y, y + int(dims[1])):
                                    if space_grid[xp, yp, z-1] != 0:
                                        support_area += 1
                            if support_area < 0.5 * total_area:
                                stability_valid = False
                                all_stability_valid = False

                        if not stability_valid:
                            continue

                        space_grid[x:x+int(dims[0]), y:y+int(dims[1]), z:z+int(dims[2])] = int(p_id[1:]) if p_id[1:].isdigit() else 1

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
    p1 = parent1.copy()
    p2 = parent2.copy()
    
    cut1 = random.randint(0, size-2)
    cut2 = random.randint(cut1+1, size-1)
    
    child = [None] * size
    child[cut1:cut2] = p1[cut1:cut2]
    
    mapping = {}
    for i in range(cut1, cut2):
        mapping[p1[i][0]] = p2[i][0]
    
    for i in range(size):
        if i < cut1 or i >= cut2:
            gene = p2[i]
            while gene[0] in [g[0] for g in child if g is not None]:
                if gene[0] in mapping:
                    mapped_id = mapping[gene[0]]
                    for g in p2:
                        if g[0] == mapped_id:
                            gene = g
                            break
                else:
                    for g in p1:
                        if g[0] not in [cg[0] for cg in child if cg is not None]:
                            gene = g
                            break
            child[i] = gene
    
    return child

def mutate(chromosome):
    mutated = chromosome.copy()
    mutation_type = random.choice(['swap_order', 'swap_rotation'])
    
    if mutation_type == 'swap_order' and len(mutated) >= 2:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    else:
        idx = random.randint(0, len(mutated)-1)
        old_orient = mutated[idx][1]
        new_orient = random.randint(1, 6)
        while new_orient == old_orient and len(mutated) > 0:
            new_orient = random.randint(1, 6)
        mutated[idx] = (mutated[idx][0], new_orient)
    
    return mutated

def run_genetic_algorithm(packages_data, container_data, params):
    random.seed(42)
    np.random.seed(42)
    
    packages = [Package(p['id'], p['length'], p['width'], p['height'], p['weight']) for p in packages_data]
    container = Container(container_data['length'], container_data['width'], container_data['height'], container_data['max_weight'])
    packages_dict = {p.id: p for p in packages}
    
    population_size = params.get('population_size', 50)
    generations = params.get('generations', 50)
    crossover_rate = params.get('crossover_rate', 0.8)
    mutation_rate = params.get('mutation_rate', 0.2)
    
    # Inisialisasi populasi
    population = [create_chromosome(packages) for _ in range(population_size)]
    
    best_solution = None
    best_fitness = -float('inf')
    
    for gen in range(generations):
        # Evaluasi fitness
        fitness_scores = []
        solutions = []
        
        for chrom in population:
            result = bottom_left_fill_with_fitness(chrom, container, packages_dict)
            fitness_scores.append(result['fitness'])
            solutions.append(result)
        
        # Update best solution
        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_fitness:
            best_fitness = fitness_scores[gen_best_idx]
            best_solution = solutions[gen_best_idx]
            best_solution['chromosome'] = population[gen_best_idx].copy()
        
        if gen < generations - 1:
            # =========== PERBAIKAN: Hitung alokasi offspring ===========
            target_crossover_children = int(population_size * crossover_rate)
            target_mutation_children = population_size - target_crossover_children

            target_crossover_ops = target_crossover_children  # 1 child per operasi
            target_mutation_ops = target_mutation_children
            
            offspring = []
            
            # Produksi CROSSOVER
            for _ in range(target_crossover_ops):
                p1_idx = tournament_selection(population, fitness_scores)
                p2_idx = tournament_selection(population, fitness_scores)
                while p2_idx == p1_idx:
                    p2_idx = tournament_selection(population, fitness_scores)
                child = pmx_crossover(population[p1_idx], population[p2_idx])
                offspring.append(child)
            
            # Produksi MUTASI
            for _ in range(target_mutation_ops):
                p_idx = tournament_selection(population, fitness_scores)
                child = mutate(population[p_idx])
                offspring.append(child)
            
            # =========== GABUNGKAN POPULASI LAMA + OFFSPRING ===========
            combined_population = population + offspring
            combined_fitness = fitness_scores + [bottom_left_fill_with_fitness(chrom, container, packages_dict)['fitness'] for chrom in offspring]
            
            # =========== SELEKSI N TERBAIK (DENGAN ELITISM) ===========
            sorted_indices = np.argsort(combined_fitness)[::-1]
            best_indices = sorted_indices[:population_size]
            population = [combined_population[i] for i in best_indices]
    
    return best_solution