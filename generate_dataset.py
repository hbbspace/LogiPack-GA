import random
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

class SackOnlyDatasetGenerator:
    """
    Generator dataset khusus karung goni untuk Gran Max Box
    
    Methods available:
    - 'surface': Logika luas permukaan kain (teoritis)
    - 'width_ratio': Logika konstanta lebar (teoritis)
    - 'fixed_dimensions': Logika dimensi fix realistik (RECOMMENDED - DEFAULT)
    
    Semua method menggunakan persentase terisi: 90%, 75%, 50%
    (Asumsi: 90% untuk ruang ikat bagian atas karung)
    
    Volume dalam satuan cm³ untuk memudahkan pemrosesan
    """
    
    def __init__(self):
        # Spesifikasi Gran Max Box (dalam cm³ untuk konsistensi)
        self.truck = {
            'name': 'Gran Max Box',
            'length_cm': 240,
            'width_cm': 160,
            'height_cm': 130,
            'volume_cm3': 240 * 160 * 130,  # 4,992,000 cm³ ≈ 5 m³
            'max_weight_kg': 800
        }

        self.sacks_spec = {
            'besar': {'weight_max_kg': 50},
            'sedang': {'weight_max_kg': 25},
            'kecil': {'weight_max_kg': 15}
        }

        # Density range (kg/cm³) - dikonversi dari kg/m³
        # 1 m³ = 1,000,000 cm³, jadi density (kg/cm³) = density (kg/m³) / 1,000,000
        self.density_map = {
            'besar': (250 / 1_000_000, 500 / 1_000_000),
            'sedang': (200 / 1_000_000, 400 / 1_000_000),
            'kecil': (150 / 1_000_000, 300 / 1_000_000)
        }

        # Berat kosong karung (kg)
        self.empty_weight_map = {
            'besar': 0.28,
            'sedang': 0.21,
            'kecil': 0.125
        }
        
        # Dimensi kain untuk logika surface & width_ratio (cm)
        self.sack_fabric_dimensions = {
            'besar': (100, 120),   # (panjang_kain_cm, tinggi_kain_cm)
            'sedang': (80, 100),
            'kecil': (55, 85)
        }
        
        # Dimensi max realistik untuk logika fixed_dimensions (karung terisi 100%) dalam cm
        # Data dari pengamatan nyata karung goni
        self.sack_max_dimensions = {
            'besar': (60, 60, 85),  # (panjang, lebar, tinggi) dalam cm 
            'sedang': (45, 45, 80),
            'kecil': (20, 20, 45)
        }
        
        # Persentase terisi (seragam untuk semua method)
        # 90% = ruang untuk mengikat bagian atas karung
        self.variation_percentages = [0.9, 0.75, 0.5]
    

    # ===============================
    # 1. LOGIKA SURFACE AREA (TEORITIS) → Volume Maksimum
    # ===============================
    def calculate_max_volume_from_surface(self, sack_type: str) -> float:
        """
        Hitung volume maksimum TEORITIS (cm³) berdasarkan luas permukaan kain
        Mengasumsikan bentuk kubus optimal dari luas kain yang tersedia
        """
        p_kain, t_kain = self.sack_fabric_dimensions[sack_type]
        
        # Surface area (cm²)
        S = 2 * p_kain * t_kain
        
        # Hitung sisi kubus optimal (akar dari luas permukaan / 6)
        x = (S / 6) ** 0.5
        
        # Volume dalam cm³
        volume_cm3 = x ** 3
        
        return round(volume_cm3, 0)
    

    # ===============================
    # 1. LOGIKA WIDTH RATIO (Konstanta Lebar/Panjang) → Volume Maksimum
    # ===============================
    def calculate_max_volume_from_width_ratio(self, sack_type: str, width_ratio: float = None) -> Tuple[float, float]:
        """
        Hitung volume maksimum TEORITIS (cm³) berdasarkan konstanta lebar/panjang
        Asumsi: lebar = konstanta × panjang, tinggi = tinggi kain
        
        Args:
            sack_type: jenis karung ('besar', 'sedang', 'kecil')
            width_ratio: konstanta lebar/panjang (default: random 0.3-0.6)
            
        Returns:
            (volume_cm3, width_ratio_yang_dipakai)
        """
        p_kain, t_kain = self.sack_fabric_dimensions[sack_type]
        
        # Pilih konstanta lebar/panjang
        if width_ratio is None:
            width_ratio = random.uniform(0.3, 0.6)
        
        # Lebar = konstanta × panjang kain
        lebar_cm = p_kain * width_ratio
        
        # Tinggi tetap menggunakan tinggi kain
        tinggi_cm = t_kain
        
        # Volume dalam cm³
        volume_cm3 = p_kain * lebar_cm * tinggi_cm
        
        return round(volume_cm3, 0), round(width_ratio, 3)
    
    
    # ===============================
    # 1. LOGIKA FIXED DIMENSIONS → Volume Maksimum
    # ===============================
    def calculate_max_volume_from_fixed(self, sack_type: str) -> float:
        """
        Hitung volume maksimum REALISTIK (cm³) dari dimensi fix
        Dimensi berdasarkan observasi karung goni sesungguhnya
        """
        p, l, t = self.sack_max_dimensions[sack_type]
        volume_cm3 = p * l * t
        return round(volume_cm3, 0)
    

    # ===============================
    # 2. Volume → Dimensi
    # ===============================
    def generate_dimensions_from_volume(self, target_volume_cm3: float) -> Tuple[float, float, float, float]:
        """
        Generate dimensi dari volume dengan rasio random yang realistis
        
        Args:
            target_volume_cm3: volume target dalam cm³
            
        Returns:
            (panjang, lebar, tinggi, actual_volume_cm3)
            
        Catatan: Rumus ini menjamin bahwa p × l × t = target_volume_cm3
        """
        # Generate rasio acak untuk bentuk yang bervariasi
        r1 = random.uniform(0.8, 1.2)  # rasio panjang vs sisi ideal
        r2 = random.uniform(0.8, 1.2)  # rasio lebar vs sisi ideal
        
        # Sisi ideal (kubus sempurna)
        x = target_volume_cm3 ** (1/3)
        
        # Dimensi dengan rasio
        p = x * r1
        l = x * r2
        t = x / (r1 * r2)
        
        actual_volume = p * l * t
        
        return round(p, 1), round(l, 1), round(t, 1), round(actual_volume, 0)
    
    # ===============================
    # 2. SCALE DIMENSIONS → Dimensi
    # ===============================
    def generate_dimensions_from_scale_dimensions(self, target_volume_cm3: float, sack_type: str) -> Tuple[float, float, float, float]:
        """
        Generate dimensi dari target volume dengan BASE = dimensi max realistik
        
        Args:
            target_volume_cm3: volume target dalam cm³
            sack_type: jenis karung ('besar', 'sedang', 'kecil')
            
        Returns:
            (panjang, lebar, tinggi, actual_volume_cm3)
            
        Logika:
            1. Dapatkan dimensi max realistik (p_max, l_max, t_max)
            2. Hitung volume max = p_max × l_max × t_max
            3. Hitung scale factor dasar = (target_volume / volume_max) ^ (1/3)
            4. Hitung dimensi dasar = (p_max × scale, l_max × scale, t_max × scale)
            5. Tambah variasi random (sama seperti generate_dimensions_from_volume)
            6. Sesuaikan dimensi ketiga agar volume = target_volume
        """
        p_max, l_max, t_max = self.sack_max_dimensions[sack_type]
        volume_max_cm3 = p_max * l_max * t_max
        
        # Hitung scale factor dasar untuk mencapai target volume
        base_scale = (target_volume_cm3 / volume_max_cm3) ** (1/3)
        
        # Dimensi dasar setelah scaling proporsional
        p_base = p_max * base_scale
        l_base = l_max * base_scale
        t_base = t_max * base_scale
        
        # Sama seperti generate_dimensions_from_volume(), tambah variasi random
        # Tapi kali ini variasi diterapkan ke p_base dan l_base
        r1 = random.uniform(0.8, 1.2)
        r2 = random.uniform(0.8, 1.2)
        
        p = p_base * r1
        l = l_base * r2
        t = target_volume_cm3 / (p * l)
        
        actual_volume = p * l * t
        
        return round(p, 1), round(l, 1), round(t, 1), round(actual_volume, 0)
    

    # ===============================
    # 3. Volume → Berat (Density-based)
    # ===============================
    def calculate_weight(self, sack_type: str, volume_cm3: float) -> float:
        """
        Hitung berat berdasarkan volume dan rentang realistis untuk ekspedisi
        
        Volume besar → berat cenderung lebih besar
        """
        weight_ranges = {
            'kecil': (7, 17),
            'sedang': (20, 40),
            'besar': (30, 60)
        }
        
        min_w, max_w = weight_ranges[sack_type]
        
        # Dapatkan volume max untuk jenis karung ini
        p, l, t = self.sack_max_dimensions[sack_type]
        max_volume_cm3 = p * l * t
        
        # Hitung rasio volume (0.5, 0.75, 0.9)
        volume_ratio = volume_cm3 / max_volume_cm3
        
        # Berat linear terhadap volume
        # volume_ratio=0.5 → berat mendekati min_w
        # volume_ratio=0.9 → berat mendekati max_w
        base_weight = min_w + (max_w - min_w) * volume_ratio
        
        # Tambah variasi random ±20% (agar berat bervariasi walau volume sama)
        variation = random.uniform(0.8, 1.2)
        final_weight = base_weight * variation
        
        # Clamp ke batas min/max
        final_weight = min(max(final_weight, min_w), max_w)
        
        return round(final_weight, 2)

    # ===============================
    # 4. Generate 1 Sack
    # ===============================
    def generate_single_sack(self, sack_type: str, variation: float, counter: Dict, method: str = 'fixed_dimensions'):
        """
        Menghasilkan satu kantong dengan spesifikasi
        
        Args:
            sack_type: jenis karung ('besar', 'sedang', 'kecil')
            variation: persentase isi (0.5, 0.75, atau 0.9)
            counter: counter untuk ID unik
            method: 'surface', 'width_ratio', atau 'fixed_dimensions'
        """
        # SEMUA method mengikuti pola yang SAMA:
        # Step 1. Hitung max_volume (berbeda cara per method)
        # Step 2. Hitung target_volume = max_volume × variation
        # Step 3. Generate dimensi DARI target_volume (tapi dengan logika berbeda)

        if method == 'surface':
            max_volume_cm3 = self.calculate_max_volume_from_surface(sack_type)
            target_volume_cm3 = max_volume_cm3 * variation
            width_ratio = None
            p, l, t, actual_volume_cm3 = self.generate_dimensions_from_volume(target_volume_cm3)
        elif method == 'width_ratio':
            max_volume_cm3, width_ratio = self.calculate_max_volume_from_width_ratio(sack_type)
            target_volume_cm3 = max_volume_cm3 * variation
            p, l, t, actual_volume_cm3 = self.generate_dimensions_from_volume(target_volume_cm3)
        else:  # fixed_dimensions (default)
            max_volume_cm3 = self.calculate_max_volume_from_fixed(sack_type)
            target_volume_cm3 = max_volume_cm3 * variation
            width_ratio = None
            p, l, t, actual_volume_cm3 = self.generate_dimensions_from_scale_dimensions(target_volume_cm3, sack_type)
        
        # Step 4: Hitung berat
        weight = self.calculate_weight(sack_type, actual_volume_cm3)
        
        # Step 5: Buat ID dan return
        counter[sack_type] += 1
        
        sack_data = {
            'id': f'SACK_{sack_type.upper()}_{counter[sack_type]:04d}',
            'type': sack_type,
            'variation_percent': int(variation * 100),
            'length_cm': round(p, 1),
            'width_cm': round(l, 1),
            'height_cm': round(t, 1),
            'volume_cm3': int(actual_volume_cm3),
            'weight_kg': weight,
            'max_possible_volume_cm3': int(max_volume_cm3),
            'method': method
        }
        
        # Tambahkan width_ratio jika method width_ratio
        if method == 'width_ratio' and width_ratio:
            sack_data['width_ratio'] = width_ratio
        
        return sack_data
    
    # ===============================
    # 5. Generate Dataset
    # ===============================
    def generate_dataset(self, method: str = 'fixed_dimensions'):
        """
        Generate dataset otomatis dengan method yang dipilih
        
        Args:
            method: 'surface', 'width_ratio', atau 'fixed_dimensions' (default)
            
        Returns:
            items: list of dictionary (data karung)
            stats: dictionary (statistik pengiriman)
        """
        items = []
        counter = {'besar': 0, 'sedang': 0, 'kecil': 0}
        
        # Distribusi jenis karung (probabilitas)
        sack_types = ['besar', 'sedang', 'kecil']
        type_weights = [0.25, 0.45, 0.30]  # 25% besar, 45% sedang, 30% kecil
        
        total_weight = 0
        max_weight = self.truck['max_weight_kg']
        max_attempts = 1000  # Mencegah infinite loop
        attempts = 0
        
        # Isi truck sampai penuh (mendekati max weight)
        while total_weight < max_weight and attempts < max_attempts:
            # Pilih jenis karung random berdasarkan weight
            sack_type = random.choices(sack_types, weights=type_weights)[0]
            
            # Pilih persentase terisi random (90%, 75%, atau 50%)
            variation = random.choice(self.variation_percentages)
            
            # Generate satu karung
            sack = self.generate_single_sack(sack_type, variation, counter, method=method)
            
            # Cek apakah muat (tidak melebihi max weight)
            if total_weight + sack['weight_kg'] <= max_weight:
                items.append(sack)
                total_weight += sack['weight_kg']
            
            attempts += 1
        
        # Hitung statistik
        total_volume_cm3 = sum(i['volume_cm3'] for i in items)
        truck_volume_cm3 = self.truck['volume_cm3']
        
        stats = {
            'total_items': len(items),
            'total_volume_cm3': total_volume_cm3,
            'total_volume_m3': round(total_volume_cm3 / 1_000_000, 2),
            'total_weight_kg': round(total_weight, 1),
            'volume_utilization_percent': round(total_volume_cm3 / truck_volume_cm3 * 100, 1),
            'weight_utilization_percent': round(total_weight / max_weight * 100, 1),
            'method_used': method,
            'truck_spec': self.truck
        }
        
        return items, stats
    
    # ===============================
    # 6. SAVE TO FILE
    # ===============================
    def save_to_csv(self, items, filename='dataset.csv'):
        """Simpan dataset ke file CSV"""
        if not items:
            print(f"⚠️ Warning: Tidak ada data untuk disimpan ke {filename}")
            return
        
        keys = items[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(items)
        print(f"✅ CSV saved to {filename} ({len(items)} records)")

    def save_to_json(self, items, stats, filename='dataset.json'):
        """Simpan dataset ke file JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_records': len(items)
                },
                'statistics': stats,
                'items': items
            }, f, indent=2)
        print(f"✅ JSON saved to {filename}")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    random.seed(42)
    
    # Inisialisasi generator
    generator = SackOnlyDatasetGenerator()
    
    # Pilih method (default: fixed_dimensions)
    # Opsi: 'fixed_dimensions', 'surface', 'width_ratio'
    METHOD = 'fixed_dimensions'
    
    print("="*40)
    print(f"GENERATOR DATASET KARUNG GONI - GRAN MAX BOX")
    print(f"Method: {METHOD.upper()}")
    print("="*40)
    print(f"📦 Truck Capacity:")
    print(f"   - Dimensions: {generator.truck['length_cm']} x {generator.truck['width_cm']} x {generator.truck['height_cm']} cm")
    print(f"   - Volume: {generator.truck['volume_cm3']:,} cm³ ({generator.truck['volume_cm3']/1_000_000} m³)")
    print(f"   - Max Weight: {generator.truck['max_weight_kg']} kg")
    print("-"*40)
    
    # Generate dataset
    items, stats = generator.generate_dataset(method=METHOD)
    
    # Tampilkan hasil
    print(f"\n📊 HASIL GENERASI:")
    print(f"   - Total Items: {stats['total_items']} karung")
    print(f"   - Total Volume: {stats['total_volume_cm3']:,} cm³ ({stats['total_volume_m3']} m³)")
    print(f"   - Total Weight: {stats['total_weight_kg']} kg")
    print(f"   - Volume Utilization: {stats['volume_utilization_percent']}%")
    print(f"   - Weight Utilization: {stats['weight_utilization_percent']}%")
    
    # Simpan ke file
    print("\n" + "-"*40)
    generator.save_to_csv(items, 'dataset.csv')
    generator.save_to_json(items, stats, 'dataset.json')
    print("="*40)
    print("✅ SELESAI! Dataset siap digunakan.")