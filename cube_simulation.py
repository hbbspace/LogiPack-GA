import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QDoubleSpinBox, QCheckBox, 
                             QPushButton, QGroupBox, QLabel, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl
import pyqtgraph as pg

# Volume konstan
V0 = 18 * 17 * 60  # cm^3

class Balok3D(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Dimensi awal
        self.p = 18  # panjang
        self.l = 17  # lebar
        self.t = 60  # tinggi
        
        # Status aktif (True = bisa diubah user)
        self.active_p = True
        self.active_l = True
        self.active_t = True
        
        # Inisialisasi UI
        self.initUI()
        
        # Update visualisasi pertama
        self.update_visualization()
        
    def initUI(self):
        self.setWindowTitle('Simulasi Balok 3D - Kontrol Dimensi')
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget utama
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout utama horizontal
        main_layout = QHBoxLayout(central_widget)
        
        # === Panel kontrol kanan ===
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(350)
        control_panel.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-size: 12px;
            }
            QDoubleSpinBox, QSlider {
                min-height: 25px;
            }
        """)
        
        # Title
        title_label = QLabel("KONTROL DIMENSI BALOK")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title_label)
        
        # Informasi volume
        self.volume_label = QLabel()
        self.volume_label.setFont(QFont("Arial", 12))
        self.volume_label.setAlignment(Qt.AlignCenter)
        self.volume_label.setStyleSheet("background-color: #e0e0e0; padding: 10px; border-radius: 5px;")
        control_layout.addWidget(self.volume_label)
        
        # === Panel Panjang ===
        p_group = QGroupBox("Dimensi Panjang (p)")
        p_layout = QVBoxLayout()
        
        # Checkbox
        self.check_p = QCheckBox("Aktif (bisa diubah)")
        self.check_p.setChecked(self.active_p)
        self.check_p.toggled.connect(self.toggle_p)
        p_layout.addWidget(self.check_p)
        
        # Slider
        self.slider_p = QSlider(Qt.Horizontal)
        self.slider_p.setRange(1, 120)
        self.slider_p.setValue(int(self.p))
        self.slider_p.valueChanged.connect(self.update_from_slider_p)
        p_layout.addWidget(self.slider_p)
        
        # Spinbox (input angka)
        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel("Nilai (cm):"))
        self.spin_p = QDoubleSpinBox()
        self.spin_p.setRange(0.1, 120)
        self.spin_p.setSingleStep(0.5)
        self.spin_p.setValue(self.p)
        self.spin_p.valueChanged.connect(self.update_from_spin_p)
        spin_layout.addWidget(self.spin_p)
        p_layout.addLayout(spin_layout)
        
        p_group.setLayout(p_layout)
        control_layout.addWidget(p_group)
        
        # === Panel Lebar ===
        l_group = QGroupBox("Dimensi Lebar (l)")
        l_layout = QVBoxLayout()
        
        self.check_l = QCheckBox("Aktif (bisa diubah)")
        self.check_l.setChecked(self.active_l)
        self.check_l.toggled.connect(self.toggle_l)
        l_layout.addWidget(self.check_l)
        
        self.slider_l = QSlider(Qt.Horizontal)
        self.slider_l.setRange(1, 120)
        self.slider_l.setValue(int(self.l))
        self.slider_l.valueChanged.connect(self.update_from_slider_l)
        l_layout.addWidget(self.slider_l)
        
        spin_layout_l = QHBoxLayout()
        spin_layout_l.addWidget(QLabel("Nilai (cm):"))
        self.spin_l = QDoubleSpinBox()
        self.spin_l.setRange(0.1, 120)
        self.spin_l.setSingleStep(0.5)
        self.spin_l.setValue(self.l)
        self.spin_l.valueChanged.connect(self.update_from_spin_l)
        spin_layout_l.addWidget(self.spin_l)
        l_layout.addLayout(spin_layout_l)
        
        l_group.setLayout(l_layout)
        control_layout.addWidget(l_group)
        
        # === Panel Tinggi ===
        t_group = QGroupBox("Dimensi Tinggi (t)")
        t_layout = QVBoxLayout()
        
        self.check_t = QCheckBox("Aktif (bisa diubah)")
        self.check_t.setChecked(self.active_t)
        self.check_t.toggled.connect(self.toggle_t)
        t_layout.addWidget(self.check_t)
        
        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setRange(1, 120)
        self.slider_t.setValue(int(self.t))
        self.slider_t.valueChanged.connect(self.update_from_slider_t)
        t_layout.addWidget(self.slider_t)
        
        spin_layout_t = QHBoxLayout()
        spin_layout_t.addWidget(QLabel("Nilai (cm):"))
        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.1, 120)
        self.spin_t.setSingleStep(0.5)
        self.spin_t.setValue(self.t)
        self.spin_t.valueChanged.connect(self.update_from_spin_t)
        spin_layout_t.addWidget(self.spin_t)
        t_layout.addLayout(spin_layout_t)
        
        t_group.setLayout(t_layout)
        control_layout.addWidget(t_group)
        
        # === Tombol Reset ===
        reset_btn = QPushButton("Reset ke Awal")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        reset_btn.clicked.connect(self.reset_dims)
        control_layout.addWidget(reset_btn)
        
        # === Informasi aturan ===
        info_group = QGroupBox("Aturan Penggunaan")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "📌 PRINSIP DASAR:\n"
            "   Volume tetap \n\n"
            "🎮 CARA MENGGUNAKAN:\n"
            "   1. CENTANG checkbox = dimensi AKTIF (bisa diubah)\n"
            "   2. KOSONGKAN = dimensi NONAKTIF (otomatis menyesuaikan)\n"
            "   3. Minimal 1 AKTIF, maksimal 2 AKTIF\n\n"
            "💡 CONTOH:\n"
            "   • Centang 'Panjang' & 'Lebar' \n"
            "     → Tinggi otomatis menyesuaikan\n"
            "   • Centang 'Panjang' saja\n"
            "     → Lebar & Tinggi otomatis menyesuaikan"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("font-size: 11px; color: #555;")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        control_layout.addWidget(info_group)
        
        control_layout.addStretch()
        
        # === Area visualisasi 3D ===
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('w')
        self.gl_widget.setCameraPosition(distance=100, elevation=45, azimuth=45)
        
        # Setup grid
        grid = gl.GLGridItem()
        grid.setSize(120, 120)
        grid.setSpacing(5, 5)
        grid.setColor((200, 200, 200, 255))
        self.gl_widget.addItem(grid)
        
        # Sumbu koordinat
        ax = gl.GLAxisItem()
        ax.setSize(120, 120, 120)
        self.gl_widget.addItem(ax)
        
        main_layout.addWidget(self.gl_widget, 2)
        main_layout.addWidget(control_panel, 1)
        
        # Update status kontrol awal
        self.update_controls_enabled()
        self.update_volume_info()
        
    def update_controls_enabled(self):
        """Enable/disable kontrol berdasarkan status aktif"""
        # Slider dan spinbox untuk panjang
        self.slider_p.setEnabled(self.active_p)
        self.spin_p.setEnabled(self.active_p)
        
        # Slider dan spinbox untuk lebar
        self.slider_l.setEnabled(self.active_l)
        self.spin_l.setEnabled(self.active_l)
        
        # Slider dan spinbox untuk tinggi
        self.slider_t.setEnabled(self.active_t)
        self.spin_t.setEnabled(self.active_t)
        
    def update_volume_info(self):
        """Update label volume"""
        current_vol = self.p * self.l * self.t
        self.volume_label.setText(f"📦 Volume Saat Ini: {current_vol:.2f} cm³\n🎯 Target Volume: {V0} cm³")
        
        if abs(current_vol - V0) > 0.01:
            self.volume_label.setStyleSheet("background-color: #ffcccc; padding: 10px; border-radius: 5px;")
        else:
            self.volume_label.setStyleSheet("background-color: #ccffcc; padding: 10px; border-radius: 5px;")
    
    def update_proportional(self, changed_dim, new_val):
        """Update dimensi dengan mekanisme proporsional"""
        # Update dimensi yang diubah user
        if changed_dim == 'p' and self.active_p:
            self.p = new_val
        elif changed_dim == 'l' and self.active_l:
            self.l = new_val
        elif changed_dim == 't' and self.active_t:
            self.t = new_val
        else:
            return False
        
        # Identifikasi dimensi nonaktif
        inactive_dims = []
        if not self.active_p:
            inactive_dims.append('p')
        if not self.active_l:
            inactive_dims.append('l')
        if not self.active_t:
            inactive_dims.append('t')
        
        # Hitung volume saat ini
        current_vol = self.p * self.l * self.t
        
        # Sesuaikan dengan dimensi nonaktif
        if abs(current_vol - V0) > 0.001 and inactive_dims:
            scale_factor = V0 / current_vol
            
            if len(inactive_dims) == 1:
                dim = inactive_dims[0]
                if dim == 'p':
                    self.p = self.p * scale_factor
                elif dim == 'l':
                    self.l = self.l * scale_factor
                elif dim == 't':
                    self.t = self.t * scale_factor
            else:
                # Dua dimensi nonaktif: proporsional
                factor = scale_factor ** (1.0 / len(inactive_dims))
                if not self.active_p:
                    self.p = self.p * factor
                if not self.active_l:
                    self.l = self.l * factor
                if not self.active_t:
                    self.t = self.t * factor
        
        # Pastikan tidak negatif dan batasi maksimal
        self.p = min(max(self.p, 0.1), 120)
        self.l = min(max(self.l, 0.1), 120)
        self.t = min(max(self.t, 0.1), 120)
        
        # Update semua kontrol
        self.update_all_controls()
        return True
    
    def update_all_controls(self):
        """Update semua kontrol dengan nilai terbaru"""
        # Blok sinyal sementara untuk menghindari loop
        self.slider_p.blockSignals(True)
        self.slider_l.blockSignals(True)
        self.slider_t.blockSignals(True)
        self.spin_p.blockSignals(True)
        self.spin_l.blockSignals(True)
        self.spin_t.blockSignals(True)
        
        # Update nilai slider (pembulatan ke integer karena slider integer)
        self.slider_p.setValue(int(round(self.p)))
        self.slider_l.setValue(int(round(self.l)))
        self.slider_t.setValue(int(round(self.t)))
        
        # Update spinbox dengan nilai presisi
        self.spin_p.setValue(self.p)
        self.spin_l.setValue(self.l)
        self.spin_t.setValue(self.t)
        
        self.slider_p.blockSignals(False)
        self.slider_l.blockSignals(False)
        self.slider_t.blockSignals(False)
        self.spin_p.blockSignals(False)
        self.spin_l.blockSignals(False)
        self.spin_t.blockSignals(False)
        
        self.update_visualization()
        self.update_volume_info()
    
    def update_visualization(self):
        """Update tampilan 3D balok"""
        # Hapus item balok lama (simpan grid dan axis)
        items_to_remove = []
        for item in self.gl_widget.items:
            if isinstance(item, gl.GLMeshItem) and item not in [self.gl_widget.items[0], self.gl_widget.items[1]]:
                items_to_remove.append(item)
        
        for item in items_to_remove:
            self.gl_widget.removeItem(item)
        
        # Buat vertices balok
        vertices = np.array([
            [0, 0, 0], [self.p, 0, 0], [self.p, self.l, 0], [0, self.l, 0],  # bawah
            [0, 0, self.t], [self.p, 0, self.t], [self.p, self.l, self.t], [0, self.l, self.t]  # atas
        ])
        
        # Definisikan faces (6 sisi)
        faces = np.array([
            [0, 1, 2, 3],  # bawah
            [4, 5, 6, 7],  # atas
            [0, 1, 5, 4],  # depan (panjang)
            [2, 3, 7, 6],  # belakang (panjang)
            [1, 2, 6, 5],  # kanan (lebar)
            [0, 3, 7, 4]   # kiri (lebar)
        ])
        
        # Warna berdasarkan status aktif
        colors = [
            (0.6, 0.8, 1.0, 0.7) if self.active_t else (0.8, 0.8, 0.8, 0.5),  # bawah (tinggi)
            (0.6, 0.8, 1.0, 0.7) if self.active_t else (0.8, 0.8, 0.8, 0.5),  # atas (tinggi)
            (1.0, 0.6, 0.6, 0.7) if self.active_p else (0.8, 0.8, 0.8, 0.5),  # depan (panjang)
            (1.0, 0.6, 0.6, 0.7) if self.active_p else (0.8, 0.8, 0.8, 0.5),  # belakang (panjang)
            (0.6, 1.0, 0.6, 0.7) if self.active_l else (0.8, 0.8, 0.8, 0.5),  # kanan (lebar)
            (0.6, 1.0, 0.6, 0.7) if self.active_l else (0.8, 0.8, 0.8, 0.5)   # kiri (lebar)
        ]
        
        # Buat mesh untuk setiap sisi
        for i, face in enumerate(faces):
            # Buat mesh data untuk face
            face_vertices = vertices[face]
            # Buat triangulasi untuk quad face (membagi jadi 2 segitiga)
            triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ])
            
            mesh_data = gl.MeshData(vertexes=face_vertices, faces=triangles)
            mesh_item = gl.GLMeshItem(meshdata=mesh_data, color=colors[i], smooth=False, 
                                      drawEdges=True, edgeColor=(0, 0, 0, 1))
            self.gl_widget.addItem(mesh_item)
        
        # Update posisi kamera
        center_x = self.p / 2
        center_y = self.l / 2
        center_z = self.t / 2
        distance = max(self.p, self.l, self.t) * 1.8
        
        # Update view
        self.gl_widget.setCameraPosition(pos=pg.Vector(center_x, center_y, center_z + distance),
                                         distance=distance)
    
    # Handler untuk update dari kontrol
    def update_from_slider_p(self, value):
        if self.active_p:
            self.update_proportional('p', float(value))
    
    def update_from_slider_l(self, value):
        if self.active_l:
            self.update_proportional('l', float(value))
    
    def update_from_slider_t(self, value):
        if self.active_t:
            self.update_proportional('t', float(value))
    
    def update_from_spin_p(self, value):
        if self.active_p:
            self.update_proportional('p', value)
    
    def update_from_spin_l(self, value):
        if self.active_l:
            self.update_proportional('l', value)
    
    def update_from_spin_t(self, value):
        if self.active_t:
            self.update_proportional('t', value)
    
    def validate_checkbox_state(self):
        """Pastikan tidak semua aktif dan tidak semua nonaktif"""
        active_count = sum([self.active_p, self.active_l, self.active_t])
        
        if active_count == 3:
            # Semua aktif -> nonaktifkan tinggi
            self.active_t = False
            self.check_t.setChecked(False)
            QMessageBox.warning(self, "Peringatan", 
                               "Tidak boleh semua dimensi AKTIF!\nTinggi dinonaktifkan.")
            return False
        elif active_count == 0:
            # Semua nonaktif -> aktifkan panjang
            self.active_p = True
            self.check_p.setChecked(True)
            QMessageBox.warning(self, "Peringatan",
                               "Tidak boleh semua dimensi NONAKTIF!\nPanjang diaktifkan.")
            return False
        return True
    
    def toggle_p(self, checked):
        self.active_p = checked
        if self.validate_checkbox_state():
            self.update_controls_enabled()
            self.update_visualization()
            # Jika menjadi nonaktif, lakukan penyesuaian volume
            if not checked:
                self.update_proportional('p', self.p)
    
    def toggle_l(self, checked):
        self.active_l = checked
        if self.validate_checkbox_state():
            self.update_controls_enabled()
            self.update_visualization()
            if not checked:
                self.update_proportional('l', self.l)
    
    def toggle_t(self, checked):
        self.active_t = checked
        if self.validate_checkbox_state():
            self.update_controls_enabled()
            self.update_visualization()
            if not checked:
                self.update_proportional('t', self.t)
    
    def reset_dims(self):
        self.p, self.l, self.t = 39.5, 39.5, 39.5
        self.active_p = True
        self.active_l = True
        self.active_t = True
        
        # Update checkbox
        self.check_p.setChecked(True)
        self.check_l.setChecked(True)
        self.check_t.setChecked(True)
        
        # Update kontrol
        self.update_controls_enabled()
        self.update_all_controls()
        
        QMessageBox.information(self, "Reset", "Dimensi direset ke awal")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = Balok3D()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()