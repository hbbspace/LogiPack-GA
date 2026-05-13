import random
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

class SackOnlyDatasetGenerator:
    """
    Generator dataset khusus karung goni untuk Gran Max Box
    """
    
    def __init__(self):
        # Spesifikasi Gran Max Box
        self.truck = {
            'name': 'Gran Max Box',
            'length_cm': 240,
            'width_cm': 160,
            'height_cm': 130,
            'volume_m3': 5,
            'max_weight_kg': 800
        }

        self.sacks_spec = {
            'besar': {'weight_max_kg': 50},
            'sedang': {'weight_max_kg': 25},
            'kecil': {'weight_max_kg': 15}
        }

        # Density range (kg/m³)
        self.density_map = {
            'besar': (250, 500),
            'sedang': (200, 400),
            'kecil': (150, 300)
        }

        # Berat kosong karung
        self.empty_weight_map = {
            'besar': 0.28,
            'sedang': 0.21,
            'kecil': 0.125
        }

        self.sack_dimensions = {
            'besar': (100, 120),   # (panjang_kain_cm, tinggi_kain_cm)
            'sedang': (80, 100),
            'kecil': (55, 85)
        }
    

    # ===============================
    # 1. LOGIKA AWAL (Surface → Volume Maksimum)
    # ===============================
    def calculate_max_volume_from_sack(self, sack_type: str) -> float:
        """
        Hitung volume maksimum berdasarkan luas kain (approach ilmiah)
        """
        # Dimensi karung kosong (kain)
        p_kain, t_kain = self.sack_dimensions[sack_type]
        
        # Surface area (cm²)
        S = 2 * p_kain * t_kain
        
        # Hitung sisi kubus optimal
        x = (S / 6) ** 0.5
        
        # Volume dalam m³
        volume_m3 = (x ** 3) / 1_000_000
        
        return round(volume_m3, 4)
    

    # ===============================
    # 1. LOGIKA BARU (Width Ratio)
    # ===============================
    def calculate_max_volume_from_sack_v2(self, sack_type: str, width_ratio: float = None) -> Tuple[float, float]:
        """
        Hitung volume maksimum berdasarkan konstanta lebar/panjang
        
        Args:
            sack_type: jenis karung ('besar', 'sedang', 'kecil')
            width_ratio: konstanta lebar/panjang (default: random 0.3-0.6)
            
        Returns:
            (volume_m3, width_ratio_yang_dipakai)
        """
        p_kain, t_kain = self.sack_dimensions[sack_type]
        
        # Pilih konstanta lebar/panjang
        if width_ratio is None:
            width_ratio = random.uniform(0.3, 0.6)
        
        # Lebar = konstanta × panjang kain
        lebar_cm = p_kain * width_ratio
        
        # Tinggi tetap menggunakan tinggi kain (karena karung berdiri)
        tinggi_cm = t_kain
        
        # Volume maksimum (cm³ → m³)
        volume_m3 = (p_kain * lebar_cm * tinggi_cm) / 1_000_000
        
        return round(volume_m3, 4), round(width_ratio, 3)
    

    # ===============================
    # 2. Volume → Dimensi
    # ===============================
    def generate_dimensions_from_volume(self, target_volume_m3: float) -> Tuple[float, float, float]:
        """
        Generate dimensi dari volume dengan rasio realistis
        """
        V = target_volume_m3 * 1_000_000  # ke cm³
        
        # Generate rasio acak
        r1 = random.uniform(0.8, 1.2)
        r2 = random.uniform(0.8, 1.2)
        
        # Misal:
        # p = x * r1
        # l = x * r2
        # t = x / (r1*r2)
        
        x = (V / (r1 * r2)) ** (1/3)
        
        p = x * r1
        l = x * r2
        t = x / (r1 * r2)
        
        return round(p, 1), round(l, 1), round(t, 1)
    

    def calculate_volume(self, p, l, t):
        """Menghitung volume dalam m³ dari dimensi cm"""
        return round((p * l * t) / 1_000_000, 4)

    # ===============================
    # 3. Volume → Berat (Density-based)
    # ===============================
    def calculate_weight(self, sack_type: str, volume_m3: float) -> float:
        density_range = self.density_map[sack_type]
        density = random.uniform(*density_range)

        content_weight = density * volume_m3
        empty_weight = self.empty_weight_map[sack_type]

        total_weight = content_weight + empty_weight

        # Clamp ke max weight
        max_weight = self.sacks_spec[sack_type]['weight_max_kg']
        return round(min(total_weight, max_weight), 2)

    # ===============================
    # 4. Generate 1 Sack
    # ===============================
    def generate_single_sack(self, sack_type: str, variation: float, counter: Dict, method: str = 'surface'):
        """
        Menghasilkan satu kantong dengan spesifikasi
        
        Args:
            sack_type: jenis karung
            variation: persentase isi (1.0, 0.75, 0.5)
            counter: counter untuk ID
            method: 'surface' (logika luas permukaan) atau 'width_ratio' (logika konstanta lebar)
        """
        if method == 'width_ratio':
            max_volume, width_ratio = self.calculate_max_volume_from_sack_v2(sack_type)
        else:  # default 'surface'
            max_volume = self.calculate_max_volume_from_sack(sack_type)
            width_ratio = None
        
        target_volume = max_volume * variation
        
        p, l, t = self.generate_dimensions_from_volume(target_volume)
        actual_volume = self.calculate_volume(p, l, t)
        
        weight = self.calculate_weight(sack_type, actual_volume)
        
        counter[sack_type] += 1
        
        sack_data = {
            'id': f'SACK_{sack_type.upper()}_{counter[sack_type]:04d}',
            'type': sack_type,
            'variation_percent': int(variation * 100),
            'length_cm': p,
            'width_cm': l,
            'height_cm': t,
            'volume_m3': actual_volume,
            'weight_kg': weight,
            'max_possible_volume_m3': max_volume,
            'method': method
        }
        
        if method == 'width_ratio' and width_ratio:
            sack_data['width_ratio'] = width_ratio
        
        return sack_data
    
    # ===============================
    # 5. Generate Dataset
    # ===============================
    def generate_dataset_auto(self, method: str = 'surface'):
        """
        Generate dataset otomatis
        
        Args:
            method: 'surface' (logika luas permukaan) atau 'width_ratio' (logika konstanta lebar)
        """
        items = []
        counter = {'besar': 0, 'sedang': 0, 'kecil': 0}
        
        sack_types = ['besar', 'sedang', 'kecil']
        type_weights = [0.25, 0.45, 0.30]
        
        variation_map = {
            'besar': [1.0, 0.75, 0.5],
            'sedang': [1.0, 0.75],
            'kecil': [1.0, 0.75]
        }
        
        total_weight = 0
        max_weight = self.truck['max_weight_kg']
        
        while total_weight < max_weight:
            sack_type = random.choices(sack_types, weights=type_weights)[0]
            variation = random.choice(variation_map[sack_type])
            
            sack = self.generate_single_sack(sack_type, variation, counter, method=method)
            
            if total_weight + sack['weight_kg'] <= max_weight:
                items.append(sack)
                total_weight += sack['weight_kg']
            else:
                break
        
        total_volume = sum(i['volume_m3'] for i in items)
        
        stats = {
            'total_items': len(items),
            'total_volume_m3': round(total_volume, 2),
            'total_weight_kg': round(total_weight, 1),
            'volume_utilization': round(total_volume / self.truck['volume_m3'] * 100, 1),
            'weight_utilization': round(total_weight / max_weight * 100, 1),
            'method_used': method
        }
        
        return items, stats
    
    # ===============================
    # SAVE
    # ===============================
    def save_to_csv(self, items, filename='dataset.csv'):
        keys = items[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(items)

    def save_to_json(self, items, stats, filename='dataset.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'items': items, 'stats': stats}, f, indent=2)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    random.seed(42)

    gen = SackOnlyDatasetGenerator()
    items, stats = gen.generate_dataset_auto()

    # Contoh penggunaan method surface (original)
    print("\n>>> MENGGUNAKAN METHOD SURFACE (ORIGINAL)")
    items_surface, stats_surface = gen.generate_dataset_auto(method='surface')
    print(f"Total Items: {stats_surface['total_items']}")
    print(f"Volume Utilization: {stats_surface['volume_utilization']}%")
    print(f"Weight Utilization: {stats_surface['weight_utilization']}%")
    print(f"Method Used: {stats_surface['method_used']}")

    # Contoh penggunaan method width_ratio (baru)
    # print("\n>>> MENGGUNAKAN METHOD WIDTH_RATIO (BARU)")
    # items_width, stats_width = gen.generate_dataset_auto(method='width_ratio')
    # print(f"Total Items: {stats_width['total_items']}")
    # print(f"Volume Utilization: {stats_width['volume_utilization']}%")
    # print(f"Weight Utilization: {stats_width['weight_utilization']}%")
    # print(f"Method Used: {stats_surface['method_used']}")

    gen.save_to_csv(items)
    gen.save_to_json(items, stats)