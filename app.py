from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from packing_engine import run_genetic_algorithm
import plotly.graph_objects as go
import os
import uuid
import json
from pathlib import Path

app = FastAPI(title="3D Bin Packing API", description="API for 3D Bin Packing using Genetic Algorithm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tentukan path ke folder public Laravel
# Asumsi struktur: 
#   parent_folder/
#   ├── logipack_vis/     (Laravel project)
#   └── logipack_ga/      (Python API project)
BASE_DIR = Path(__file__).resolve().parent  # logipack_ga/
PROJECT_ROOT = BASE_DIR.parent              # folder parent (tempat kedua proyek berada)
LARAVEL_PUBLIC_VISUALIZATIONS = PROJECT_ROOT / "LogiPack-Viz" / "public" / "visualizations"

class PackageInput(BaseModel):
    id: str
    length: float
    width: float
    height: float
    weight: float

class ContainerInput(BaseModel):
    length: float
    width: float
    height: float
    max_weight: float

class GAParams(BaseModel):
    population_size: int = 50
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2

class PackingRequest(BaseModel):
    container: ContainerInput
    packages: List[PackageInput]
    ga_params: Optional[GAParams] = None

class PackingResponse(BaseModel):
    success: bool
    fitness: float
    volume_utilization: float
    weight_utilization: float
    total_volume: float
    total_weight: float
    num_placed: int
    total_packages: int
    center_of_gravity: List[float]
    placed_packages: List[Dict[str, Any]]
    unplaced_packages: List[str]
    visualization_html: Optional[str] = None
    message: Optional[str] = None

def generate_visualization(positions, container_dims, filename):
    """
    Menghasilkan visualisasi 3D dan menyimpannya ke folder public Laravel
    
    Args:
        positions: List posisi paket
        container_dims: Tuple (panjang, lebar, tinggi) container
        filename: Nama file tanpa ekstensi
    
    Returns:
        str: Path relatif yang bisa diakses dari browser via Laravel
    """
    fig = go.Figure()
    
    # Gambar container (wireframe)
    container_vertices = [
        [0, 0, 0], [container_dims[0], 0, 0],
        [container_dims[0], container_dims[1], 0], [0, container_dims[1], 0],
        [0, 0, container_dims[2]], [container_dims[0], 0, container_dims[2]],
        [container_dims[0], container_dims[1], container_dims[2]], [0, container_dims[1], container_dims[2]]
    ]
    
    container_lines = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
    
    for line in container_lines:
        fig.add_trace(go.Scatter3d(
            x=[container_vertices[line[0]][0], container_vertices[line[1]][0]],
            y=[container_vertices[line[0]][1], container_vertices[line[1]][1]],
            z=[container_vertices[line[0]][2], container_vertices[line[1]][2]],
            mode='lines',
            line=dict(color='#FF0000', width=3),
            showlegend=False
        ))
    
    # Warna untuk paket (mengikuti warna brand Pos Indonesia)
    colors = ['#FF0000', '#0066B3', '#00AA00', '#FF6600', '#9900CC', '#FFCC00']
    
    for i, pos in enumerate(positions):
        if pos.get('placed', False):
            x, y, z = pos['x'], pos['y'], pos['z']
            dx, dy, dz = pos['dx'], pos['dy'], pos['dz']
            
            color_idx = i % len(colors)
            color = colors[color_idx]
            
            fig.add_trace(go.Mesh3d(
                x=[x, x+dx, x+dx, x, x, x+dx, x+dx, x],
                y=[y, y, y+dy, y+dy, y, y, y+dy, y+dy],
                z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
                i=[0,0,0,1,1,2],
                j=[1,2,4,3,5,6],
                k=[2,3,5,7,6,7],
                color=color,
                opacity=0.7,
                name=f"{pos['id']} ({pos['dx']}x{pos['dy']}x{pos['dz']})"
            ))
    
    fig.update_layout(
        title=dict(text=f"3D Bin Packing Result - {filename}", x=0.5),
        scene=dict(
            xaxis_title='Panjang (X) cm',
            yaxis_title='Lebar (Y) cm',
            zaxis_title='Tinggi (Z) cm',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Buat folder tujuan jika belum ada
    try:
        LARAVEL_PUBLIC_VISUALIZATIONS.mkdir(parents=True, exist_ok=True)
        print(f"📁 Folder visualisasi: {LARAVEL_PUBLIC_VISUALIZATIONS}")
    except Exception as e:
        print(f"⚠️ Gagal membuat folder: {e}")
        # Fallback: simpan di folder lokal visualizations
        local_viz_path = Path("visualizations")
        local_viz_path.mkdir(exist_ok=True)
        html_path = local_viz_path / f"{filename}.html"
        fig.write_html(str(html_path))
        print(f"⚠️ Fallback: Visualisasi disimpan di {html_path}")
        return f"/visualizations/{filename}.html"
    
    # Simpan file HTML di folder public Laravel
    html_path = LARAVEL_PUBLIC_VISUALIZATIONS / f"{filename}.html"
    
    try:
        fig.write_html(str(html_path))
        print(f"✅ Visualisasi disimpan di: {html_path}")
        print(f"🌐 URL akses: /visualizations/{filename}.html")
    except Exception as e:
        print(f"❌ Gagal menyimpan visualisasi: {e}")
        # Fallback: simpan di folder lokal
        local_viz_path = Path("visualizations")
        local_viz_path.mkdir(exist_ok=True)
        html_path = local_viz_path / f"{filename}.html"
        fig.write_html(str(html_path))
        print(f"⚠️ Fallback: Visualisasi disimpan di {html_path}")
        return f"/visualizations/{filename}.html"
    
    # Kembalikan path relatif untuk diakses dari browser
    return f"/visualizations/{filename}.html"

@app.post("/api/pack", response_model=PackingResponse)
async def pack_items(request: PackingRequest):
    try:
        if not request.packages:
            raise HTTPException(status_code=400, detail="No packages provided")
        
        # Gunakan GA params default jika tidak disediakan
        ga_params = request.ga_params if request.ga_params else GAParams()
        
        # Jalankan algoritma genetika
        result = run_genetic_algorithm(
            packages_data=[p.model_dump() for p in request.packages],
            container_data=request.container.model_dump(),
            params=ga_params.model_dump()
        )
        
        # Pisahkan paket yang terpasang dan tidak terpasang
        placed_packages = [p for p in result['positions'] if p.get('placed', False)]
        unplaced_packages = [p['id'] for p in result['positions'] if not p.get('placed', False)]
        
        # Generate visualisasi
        viz_id = str(uuid.uuid4())[:8]
        container_dims = (request.container.length, request.container.width, request.container.height)
        viz_path = generate_visualization(result['positions'], container_dims, f"packing_{viz_id}")
        
        # Siapkan response
        response_data = {
            "success": True,
            "fitness": result['fitness'],
            "volume_utilization": result['volume_utilization'],
            "weight_utilization": result['weight_utilization'],
            "total_volume": result['total_volume'],
            "total_weight": result['total_weight'],
            "num_placed": result['num_placed'],
            "total_packages": len(request.packages),
            "center_of_gravity": result['center_of_gravity'],
            "placed_packages": placed_packages,
            "unplaced_packages": unplaced_packages,
            "visualization_html": viz_path,
            "message": f"Successfully packed {result['num_placed']} out of {len(request.packages)} packages"
        }
        
        print(f"📊 Packing Result: {response_data['num_placed']}/{response_data['total_packages']} packages placed")
        print(f"📈 Volume Utilization: {response_data['volume_utilization']:.2f}%")
        print(f"🎯 Fitness Score: {response_data['fitness']:.2f}")
        
        return PackingResponse(**response_data)
    
    except Exception as e:
        print(f"❌ Error in pack_items: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "3D Bin Packing API is running",
        "visualization_path": str(LARAVEL_PUBLIC_VISUALIZATIONS)
    }

if __name__ == "__main__":
    print("🚀 Starting 3D Bin Packing API...")
    print(f"📁 Visualizations will be saved to: {LARAVEL_PUBLIC_VISUALIZATIONS}")
    print("🌐 Server running on http://0.0.0.0:8001")
    print("📡 API Docs available at http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)