import random
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

class SackOnlyDatasetGenerator:
    """
    Generator dataset khusus kantong goni untuk Truk Fuso Box
    - Hanya kantong (tanpa paket besar)
    - Dimensi random, volume sesuai variasi
    - Berat random (independen dari volume)
    - Batasan: total berat < kapasitas truk
    """
    
    def __init__(self):
        # Spesifikasi Truk Fuso Box
        self.truck = {
            'name': 'Gran Max Box',
            'length_cm': 240,
            'width_cm': 160,
            'height_cm': 130,
            'volume_m3': 5,
            'max_weight_kg': 800
        }
        
        # Spesifikasi karung goni
        self.sacks_spec = {
            'besar': {
                'weight_max_kg': 50, # 280 gram
                'volume_variations': [1.0, 0.75, 0.50],
                'dimension_ranges': {
                    'length': (80, 100),   # cm
                    'width': (40, 60), # ??
                    'height': (100, 120)
                }
            },
            'sedang': {
                'weight_max_kg': 25, # 210 gram
                'volume_variations': [1.0, 0.75],
                'dimension_ranges': {
                    'length': (60, 80),
                    'width': (30, 45), # ??
                    'height': (80, 100)
                }
            },
            'kecil': {
                'weight_max_kg': 15, # 125 gram
                'volume_variations': [1.0, 0.75],
                'dimension_ranges': {
                    'length': (35, 55),
                    'width': (25, 35), # ??
                    'height': (65, 85)
                }
            }
        }
        
        # Estimasi awal untuk jumlah kantong (akan disesuaikan otomatis)
        self.target_weight_kg = self.truck['max_weight_kg'] * 0.9  # Target 85% dari kapasitas
        
    def calculate_volume_from_dimensions(self, length: float, width: float, height: float) -> float:
        """Menghitung volume dalam m³ dari dimensi cm"""
        return round((length * width * height) / 1_000_000, 4)
    
    def generate_random_dimensions(self, sack_type: str, target_volume_m3: float, max_iterations: int = 100) -> Tuple[float, float, float]:
        """
        Menghasilkan dimensi random (balok) yang volumenya mendekati target_volume_m3
        """
        ranges = self.sacks_spec[sack_type]['dimension_ranges']
        target_volume_cm3 = target_volume_m3 * 1_000_000
        
        for _ in range(max_iterations):
            length = random.uniform(*ranges['length'])
            width = random.uniform(*ranges['width'])
            height = random.uniform(*ranges['height'])
            
            volume = length * width * height
            
            # Terima jika volume dalam range 90-110% dari target
            if 0.9 * target_volume_cm3 <= volume <= 1.1 * target_volume_cm3:
                return round(length, 1), round(width, 1), round(height, 1)
        
        # Jika tidak ketemu, scaling dari dimensi rata-rata
        avg_length = sum(ranges['length']) / 2
        avg_width = sum(ranges['width']) / 2
        avg_height = sum(ranges['height']) / 2
        avg_volume = avg_length * avg_width * avg_height
        
        scale_factor = (target_volume_cm3 / avg_volume) ** (1/3)
        
        return (
            round(avg_length * scale_factor, 1),
            round(avg_width * scale_factor, 1),
            round(avg_height * scale_factor, 1)
        )
    
    def generate_single_sack(self, sack_type: str, variation: float, sack_id: int, type_counter: Dict[str, int]) -> Dict[str, Any]:
        """
        Menghasilkan satu kantong dengan spesifikasi:
        - Volume sesuai variasi (dari volume maksimum)
        - Dimensi random (balok)
        - Berat random (0 s/d weight_max_kg)
        """
        spec = self.sacks_spec[sack_type]
        
        # Hitung volume maksimum dari dimensi range
        ranges = spec['dimension_ranges']
        max_volume_m3 = self.calculate_volume_from_dimensions(
            ranges['length'][1], ranges['width'][1], ranges['height'][1]
        )
        
        # Volume target sesuai variasi
        target_volume_m3 = round(max_volume_m3 * variation, 4)
        
        # Generate dimensi random dengan volume mendekati target
        length, width, height = self.generate_random_dimensions(sack_type, target_volume_m3)
        actual_volume = self.calculate_volume_from_dimensions(length, width, height)
        
        # Generate berat random (0 s/d weight_max_kg)
        weight_kg = round(random.uniform(0, spec['weight_max_kg']), 1)
        
        type_counter[sack_type] += 1
        
        return {
            'id': f'SACK_{sack_type.upper()}_{type_counter[sack_type]:04d}',
            'type': sack_type,
            'variation_percent': int(variation * 100),
            'length_cm': length,
            'width_cm': width,
            'height_cm': height,
            'volume_m3': actual_volume,
            'weight_kg': weight_kg,
            'max_possible_volume_m3': max_volume_m3,
            'max_possible_weight_kg': spec['weight_max_kg']
        }
    
    def generate_dataset_auto(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate dataset otomatis dengan total berat < kapasitas truk
        Volume bisa melebihi kapasitas (untuk uji optimasi)
        """
        items = []
        type_counter = {'besar': 0, 'sedang': 0, 'kecil': 0}
        
        # Probabilitas kemunculan tiap jenis kantong
        sack_types = ['besar', 'sedang', 'kecil']
        type_weights = [0.25, 0.45, 0.30]  # 25% besar, 45% sedang, 30% kecil
        
        total_weight = 0
        max_weight = self.truck['max_weight_kg']
        
        # Variasi volume untuk setiap jenis
        variations_map = {
            'besar': [1.0, 0.75, 0.50],
            'sedang': [1.0, 0.50],
            'kecil': [1.0, 0.50]
        }
        
        # Probabilitas variasi (bisa disesuaikan)
        variation_weights_map = {
            'besar': [0.4, 0.35, 0.25],  # 40% penuh, 35% 75%, 25% 50%
            'sedang': [0.6, 0.4],        # 60% penuh, 40% 50%
            'kecil': [0.55, 0.45]        # 55% penuh, 45% 50%
        }
        
        iteration = 0
        max_iterations = 500  # Safety limit
        
        while total_weight < max_weight and iteration < max_iterations:
            # Pilih jenis kantong
            sack_type = random.choices(sack_types, weights=type_weights, k=1)[0]
            
            # Pilih variasi volume
            variations = variations_map[sack_type]
            var_weights = variation_weights_map[sack_type]
            variation = random.choices(variations, weights=var_weights, k=1)[0]
            
            # Generate kantong
            new_sack = self.generate_single_sack(
                sack_type, variation, 
                len(items) + 1, type_counter
            )
            
            # Cek apakah menambah kantong ini akan melebihi kapasitas
            if total_weight + new_sack['weight_kg'] <= max_weight:
                items.append(new_sack)
                total_weight += new_sack['weight_kg']
            else:
                # Jika hampir penuh, coba generate kantong dengan berat kecil
                # Maksimal 3 kali percobaan untuk menghindari infinite loop
                for _ in range(3):
                    light_sack = self.generate_single_sack(
                        sack_type, variation, 
                        len(items) + 1, type_counter
                    )
                    # Cari yang beratnya kecil
                    if light_sack['weight_kg'] < (max_weight - total_weight):
                        items.append(light_sack)
                        total_weight += light_sack['weight_kg']
                        break
                break  # Keluar loop karena sudah mendekati kapasitas
            
            iteration += 1
        
        # Hitung statistik
        total_volume = sum(item['volume_m3'] for item in items)
        
        stats = {
            'total_items': len(items),
            'total_volume_m3': round(total_volume, 2),
            'total_weight_kg': round(total_weight, 1),
            'truck_volume_m3': self.truck['volume_m3'],
            'truck_max_weight_kg': self.truck['max_weight_kg'],
            'volume_utilization': round((total_volume / self.truck['volume_m3']) * 100, 1),
            'weight_utilization': round((total_weight / self.truck['max_weight_kg']) * 100, 1),
            'is_volume_exceed': total_volume > self.truck['volume_m3'],
            'volume_exceed_percent': round(((total_volume - self.truck['volume_m3']) / self.truck['volume_m3']) * 100, 1) if total_volume > self.truck['volume_m3'] else 0
        }
        
        # Acak urutan item
        random.shuffle(items)
        
        return items, stats
    
    def save_to_csv(self, items: List[Dict[str, Any]], filename: str = 'sacks_dataset.csv'):
        """Menyimpan dataset ke CSV"""
        if not items:
            print("Tidak ada data untuk disimpan!")
            return
        
        fieldnames = ['id', 'type', 'variation_percent', 'length_cm', 'width_cm', 
                     'height_cm', 'volume_m3', 'weight_kg', 'max_possible_volume_m3', 
                     'max_possible_weight_kg']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in items:
                row = {field: item.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"✅ Data disimpan ke {filename}")
    
    def save_to_json(self, items: List[Dict[str, Any]], stats: Dict[str, Any], filename: str = 'sacks_dataset.json'):
        """Menyimpan dataset ke JSON dengan metadata"""
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'description': 'Dataset kantong goni untuk Truk Fuso Box - Berat random, volume bisa exceed kapasitas',
                'notes': 'Volume dapat melebihi kapasitas truk untuk menguji efisiensi algoritma packing'
            },
            'truck_specifications': self.truck,
            'statistics': stats,
            'cargo_items': items
        }
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(output, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✅ Data disimpan ke {filename}")
    
    def print_summary(self, items: List[Dict[str, Any]], stats: Dict[str, Any]):
        """Mencetak ringkasan dataset"""
        print("\n" + "="*75)
        print("📦 GENERATOR KANTONG GONI UNTUK TRUK FUSO BOX")
        print("="*75)
        
        print(f"\n🚚 SPESIFIKASI TRUK:")
        print(f"   Model: {self.truck['name']}")
        print(f"   Dimensi: {self.truck['length_cm']}cm x {self.truck['width_cm']}cm x {self.truck['height_cm']}cm")
        print(f"   Volume Maks: {self.truck['volume_m3']} m³")
        print(f"   Berat Maks: {self.truck['max_weight_kg']} kg ({self.truck['max_weight_kg']/1000} ton)")
        
        print(f"\n📊 STATISTIK MUATAN (KANTONG SAJA):")
        print(f"   Total Kantong: {stats['total_items']} buah")
        print(f"   Total Volume: {stats['total_volume_m3']} m³")
        print(f"   Total Berat: {stats['total_weight_kg']} kg ({stats['total_weight_kg']/1000:.1f} ton)")
        print(f"   Utilisasi Berat: {stats['weight_utilization']}%")
        
        if stats['is_volume_exceed']:
            print(f"\n⚠️  VOLUME MELEBIHI KAPASITAS TRUK!")
            print(f"   Kelebihan volume: {stats['volume_exceed_percent']}% dari kapasitas")
            print(f"   (Ini disengaja untuk menguji algoritma packing yang efisien)")
        else:
            print(f"   Utilisasi Volume: {stats['volume_utilization']}%")
        
        # Komposisi kantong
        print(f"\n📦 KOMPOSISI KANTONG:")
        type_counts = {}
        variation_counts = {}
        
        for item in items:
            t = item['type']
            var = item['variation_percent']
            type_counts[t] = type_counts.get(t, 0) + 1
            variation_counts[f"{t}_{var}%"] = variation_counts.get(f"{t}_{var}%", 0) + 1
        
        for sack_type in ['besar', 'sedang', 'kecil']:
            count = type_counts.get(sack_type, 0)
            if count > 0:
                print(f"\n   Kantong {sack_type.capitalize()}: {count} buah")
                for var in [100, 75, 50]:
                    var_key = f"{sack_type}_{var}%"
                    if var_key in variation_counts:
                        print(f"      - Variasi {var}%: {variation_counts[var_key]} buah")
        
        # Statistik berat per jenis
        print(f"\n⚖️  STATISTIK BERAT (kg):")
        for sack_type in ['besar', 'sedang', 'kecil']:
            type_items = [i for i in items if i['type'] == sack_type]
            if type_items:
                weights = [i['weight_kg'] for i in type_items]
                print(f"   {sack_type.capitalize()}: min={min(weights):.1f}, max={max(weights):.1f}, avg={sum(weights)/len(weights):.1f}")
        
        print("\n" + "="*75)


def main():
    """Fungsi utama"""
    print("🚛 GENERATOR KANTONG GONI UNTUK TRUK FUSO BOX")
    print("="*50)
    print("\nSpesifikasi:")
    print("  ✓ Hanya kantong goni (tanpa paket besar)")
    print("  ✓ Dimensi random, volume sesuai variasi (100%, 75%, 50%)")
    print("  ✓ Berat random (0 s/d berat maks kantong)")
    print("  ✓ Batasan: total berat < kapasitas truk (10 ton)")
    print("  ✓ Volume BISA melebihi kapasitas (untuk uji optimasi)")
    print("\n" + "="*50)
    
    # Inisialisasi generator
    generator = SackOnlyDatasetGenerator()
    
    print("\n📋 Meng-generate dataset...")
    items, stats = generator.generate_dataset_auto()
    
    # Tampilkan ringkasan
    generator.print_summary(items, stats)
    
    # Simpan ke file
    generator.save_to_csv(items, 'sacks_dataset.csv')
    generator.save_to_json(items, stats, 'sacks_dataset.json')
    
    # Tampilkan contoh data (10 item pertama)
    # print("\n📋 CONTOH DATA (10 KANTONG PERTAMA):")
    # print("-" * 110)
    # print(f"{'ID':<18} {'Jenis':<8} {'Variasi':<7} {'Dimensi (cm)':<22} {'Volume(m³)':<11} {'Berat(kg)':<10}")
    # print("-" * 110)
    # for item in items[:10]:
    #     dims = f"{item['length_cm']}x{item['width_cm']}x{item['height_cm']}"
    #     print(f"{item['id']:<18} {item['type']:<8} {item['variation_percent']}%{'':<4} {dims:<22} {item['volume_m3']:<11.4f} {item['weight_kg']:<10.1f}")
    
    # if len(items) > 10:
    #     print(f"\n... dan {len(items) - 10} kantong lainnya")
    
    print("\n" + "="*50)
    print("✅ GENERASI DATASET SELESAI!")
    print("📁 File yang dihasilkan:")
    print("   - sacks_dataset.csv (data kantong dalam format CSV)")
    print("   - sacks_dataset.json (data lengkap + metadata + statistik)")
    print("="*50)


if __name__ == "__main__":
    random.seed(42)  # Bisa diubah atau dihapus untuk variasi berbeda
    main()