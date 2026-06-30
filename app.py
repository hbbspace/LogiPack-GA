from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from packing_engine import run_genetic_algorithm
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.express as px
import os
import uuid
import json
from pathlib import Path

app = FastAPI(title="3D Bin Packing API", description="API for 3D Bin Packing using Genetic Algorithm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tentukan path ke folder public Laravel
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
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
    # === TAMBAHAN BARU ===
    chromosome: Optional[List] = None
    execution_time_seconds: Optional[float] = None
    history: Optional[List[Dict]] = None


def generate_visualization(positions, container_dims, filename):
    fig = go.Figure()
    
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
            line=dict(color='gray', width=3),
            showlegend=False
        ))
    
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    placed_positions = [p for p in positions if p.get('placed', False)]
    valid_positions = [p for p in placed_positions if p['x'] >= 0]

    package_names = []
    
    for i, pos in enumerate(valid_positions):
        x, y, z = pos['x'], pos['y'], pos['z']
        dx, dy, dz = pos['dx'], pos['dy'], pos['dz']
        orientation = pos.get('orientation', pos.get('rot_index', 0))
        tracking_number = pos.get('tracking_number', pos.get('id', f'Paket_{i+1}'))
        color = colors[i % len(colors)]
        group_name = f"{tracking_number}"
        package_names.append(group_name)
        
        fig.add_trace(go.Mesh3d(
            x=[x, x+dx, x+dx, x, x, x+dx, x+dx, x],
            y=[y, y, y+dy, y+dy, y, y, y+dy, y+dy],
            z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.75,
            hovertext=f"""
            <b>Paket: {tracking_number}</b><br>
            """,
            hoverinfo="text",
            color=color,
            flatshading=True,
            name=group_name,
            showlegend=True,
            legendgroup=group_name,
        ))
        
        # --- OUTLINE HITAM TEBAL---
        # 8 titik sudut paket
        vertices = [
            (x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z),
            (x, y, z+dz), (x+dx, y, z+dz), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)
        ]
        
        # Garis tepi kotak
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # Alas
            (4,5), (5,6), (6,7), (7,4),  # Atap
            (0,4), (1,5), (2,6), (3,7)   # Tiang vertikal
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                mode='lines',
                line=dict(color='black', width=3),  # Outline hitam tebal
                showlegend=False,
                legendgroup=group_name,
                hoverinfo='skip'
            ))
        
        # --- LABEL ID PAKET + ORIENTASI DI TENGAH ---
        # Format label: ID Paket + orientasi
        orientation_map = {
            1: "LWH", 2: "LHW", 3: "WLH", 
            4: "WHL", 5: "HLW", 6: "HWL"
        }
        
        if orientation >= 1 and orientation <= 6:
            rot_text = orientation_map[orientation]
        else:
            rot_text = f"rot:{orientation}"
        
        label_text = f"<b>{tracking_number}</b><br><sub>({rot_text})</sub>"
        
        fig.add_trace(go.Scatter3d(
            x=[x + dx/2],
            y=[y + dy/2],
            z=[z + dz/2],
            mode='text',
            text=[label_text],
            textfont=dict(size=11, color="black", family="Arial Black"),
            showlegend=False,
            legendgroup=group_name,
            hoverinfo='skip'
        ))
    
    # 5. Layout akademik bersih (seperti referensi)
    total_volume = sum(p['dx']*p['dy']*p['dz'] for p in valid_positions)
    total_weight = sum(p.get('weight', 0) for p in valid_positions)
    container_volume = container_dims[0] * container_dims[1] * container_dims[2]
    utilization = (total_volume / container_volume * 100) if container_volume > 0 else 0
    
    fig.update_layout(
        title=dict(
            text=f"<b>3D Bin Packing Result - {filename}</b><br>"
                 f"Utilisasi: {round(utilization, 2)}% | Volume: {round(total_volume, 1)} cm³ | Berat: {round(total_weight, 2)} kg",
            x=0.02,
            font=dict(size=14)
        ),
        legend=dict(
            title="<b>Paket Termuat:</b>",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11, color="black")
        ),
        scene=dict(
            xaxis=dict(title="Panjang (X) cm", range=[0, container_dims[0]], showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="Lebar (Y) cm", range=[0, container_dims[1]], showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title="Tinggi (Z) cm", range=[0, container_dims[2]], showgrid=True, gridcolor='lightgray'),
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            bgcolor="white"
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    try:
        LARAVEL_PUBLIC_VISUALIZATIONS.mkdir(parents=True, exist_ok=True)
        # print(f"📁 Folder visualisasi: {LARAVEL_PUBLIC_VISUALIZATIONS}")
    except Exception as e:
        print(f"⚠️ Gagal membuat folder: {e}")
        local_viz_path = Path("visualizations")
        local_viz_path.mkdir(exist_ok=True)
        html_path = local_viz_path / f"{filename}.html"
        fig.write_html(str(html_path))
        print(f"⚠️ Fallback: Visualisasi disimpan di {html_path}")
        return f"/visualizations/{filename}.html"
    
    html_path = LARAVEL_PUBLIC_VISUALIZATIONS / f"{filename}.html"
    
    try:
        fig.write_html(str(html_path))
        print(f"✅ Visualisasi disimpan di: {html_path}")
        # print(f"🌐 URL akses: /visualizations/{filename}.html")
    except Exception as e:
        print(f"❌ Gagal menyimpan visualisasi: {e}")
        local_viz_path = Path("visualizations")
        local_viz_path.mkdir(exist_ok=True)
        html_path = local_viz_path / f"{filename}.html"
        fig.write_html(str(html_path))
        print(f"⚠️ Fallback: Visualisasi disimpan di {html_path}")
        return f"/visualizations/{filename}.html"
    
    return f"/visualizations/{filename}.html"


@app.post("/api/pack", response_model=PackingResponse)
async def pack_items(request: PackingRequest):
    try:
        if not request.packages:
            raise HTTPException(status_code=400, detail="No packages provided")
        
        ga_params = request.ga_params if request.ga_params else GAParams()
        
        print(f"📦 Processing {len(request.packages)} packages with GA params:")
        print(f"   Population: {ga_params.population_size}, Generations: {ga_params.generations}")
        
        result = run_genetic_algorithm(
            packages_data=[p.model_dump() for p in request.packages],
            container_data=request.container.model_dump(),
            params=ga_params.model_dump()
        )
        
        placed_packages = [p for p in result['positions'] if p.get('placed', False)]
        unplaced_packages = [p['id'] for p in result['positions'] if not p.get('placed', False)]
        
        viz_id = str(uuid.uuid4())[:8]
        container_dims = (request.container.length, request.container.width, request.container.height)
        viz_path = generate_visualization(result['positions'], container_dims, f"packing_{viz_id}")
        
        # === TAMBAHAN: Ambil chromosome dan history dari result ===
        chromosome = result.get('chromosome', [])
        execution_time = result.get('execution_time_seconds', 0)
        history = result.get('history', [])
        
        # Konversi chromosome ke format yang lebih mudah dibaca
        # Format asli: [("P001", 1), ("P002", 3), ...]
        formatted_chromosome = []
        for item in chromosome:
            if isinstance(item, tuple) and len(item) == 2:
                formatted_chromosome.append([item[0], item[1]])
            elif isinstance(item, list) and len(item) == 2:
                formatted_chromosome.append(item)
        
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
            "message": f"Successfully packed {result['num_placed']} out of {len(request.packages)} packages",
            # === TAMBAHAN BARU ===
            "chromosome": formatted_chromosome,
            "execution_time_seconds": execution_time,
            "history": history
        }
        
        print(f"📊 Packing Result: {response_data['num_placed']}/{response_data['total_packages']} packages placed")
        print(f"📈 Volume Utilization: {response_data['volume_utilization']:.2f}%")
        print(f"🎯 Fitness Score: {response_data['fitness']:.2f}")
        print(f"⏱️ Execution Time: {execution_time:.2f} seconds")
        print(f"📜 History records: {len(history)} generations")
        
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