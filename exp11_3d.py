import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd

@dataclass
class Item:
    """Represents a 3D item with dimensions and position"""
    id: int
    width: float
    height: float
    depth: float
    x: float = 0
    y: float = 0
    z: float = 0
    packed: bool = False
    
    @property
    def volume(self):
        return self.width * self.height * self.depth

@dataclass
class Container:
    """Represents a 3D container/bin"""
    width: float
    height: float
    depth: float
    items: List[Item] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
    
    @property
    def volume(self):
        return self.width * self.height * self.depth
    
    @property
    def used_volume(self):
        return sum(item.volume for item in self.items)
    
    @property
    def utilization(self):
        return (self.used_volume / self.volume) * 100

class BinPacker3D:
    """Simple 3D bin packing algorithm using Bottom-Left-Fill heuristic"""
    
    def __init__(self, container: Container):
        self.container = container
        self.free_spaces = [(0, 0, 0, container.width, container.height, container.depth)]
    
    def can_fit(self, item: Item, x: float, y: float, z: float) -> bool:
        """Check if item can fit at given position"""
        return (x + item.width <= self.container.width and
                y + item.height <= self.container.height and
                z + item.depth <= self.container.depth)
    
    def overlaps_with_existing(self, item: Item, x: float, y: float, z: float) -> bool:
        """Check if item overlaps with existing items"""
        for existing_item in self.container.items:
            # Two boxes overlap if they overlap in ALL three dimensions
            x_overlap = not (x >= existing_item.x + existing_item.width or x + item.width <= existing_item.x)
            y_overlap = not (y >= existing_item.y + existing_item.height or y + item.height <= existing_item.y)
            z_overlap = not (z >= existing_item.z + existing_item.depth or z + item.depth <= existing_item.z)
            
            if x_overlap and y_overlap and z_overlap:
                return True
        return False
    
    def find_position(self, item: Item) -> Optional[Tuple[float, float, float]]:
        """Find a suitable position for the item using systematic grid approach"""
        # Create a more systematic approach to avoid overlaps
        step_size = 1.0  # Grid step size for position testing
        
        # Generate candidate positions in a systematic way
        positions_to_try = []
        
        # Start with origin
        positions_to_try.append((0, 0, 0))
        
        # Add positions based on existing items (corner positions)
        for existing_item in self.container.items:
            # Right side of existing item
            if existing_item.x + existing_item.width + item.width <= self.container.width:
                positions_to_try.append((existing_item.x + existing_item.width, existing_item.y, existing_item.z))
            
            # Top of existing item
            if existing_item.y + existing_item.height + item.height <= self.container.height:
                positions_to_try.append((existing_item.x, existing_item.y + existing_item.height, existing_item.z))
            
            # Behind existing item
            if existing_item.z + existing_item.depth + item.depth <= self.container.depth:
                positions_to_try.append((existing_item.x, existing_item.y, existing_item.z + existing_item.depth))
        
        # Add systematic grid positions if needed
        for z in np.arange(0, self.container.depth - item.depth + 0.1, step_size):
            for y in np.arange(0, self.container.height - item.height + 0.1, step_size):
                for x in np.arange(0, self.container.width - item.width + 0.1, step_size):
                    positions_to_try.append((x, y, z))
        
        # Remove duplicates and sort by z, y, x (bottom-left-front preference)
        positions_to_try = list(set(positions_to_try))
        positions_to_try.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        
        for x, y, z in positions_to_try:
            if (self.can_fit(item, x, y, z) and 
                not self.overlaps_with_existing(item, x, y, z)):
                return (x, y, z)
        
        return None
    
    def pack_item(self, item: Item) -> bool:
        """Try to pack a single item"""
        position = self.find_position(item)
        if position:
            item.x, item.y, item.z = position
            item.packed = True
            self.container.items.append(item)
            return True
        return False
    
    def pack_items(self, items: List[Item]) -> Tuple[List[Item], List[Item]]:
        """Pack multiple items, return packed and unpacked lists"""
        # Sort items by volume (descending) for better packing
        sorted_items = sorted(items, key=lambda x: x.volume, reverse=True)
        
        packed_items = []
        unpacked_items = []
        
        for item in sorted_items:
            if self.pack_item(item):
                packed_items.append(item)
            else:
                unpacked_items.append(item)
        
        return packed_items, unpacked_items

def generate_random_items(num_items: int, min_dim: float = 1, max_dim: float = 10) -> List[Item]:
    """Generate random items with random dimensions"""
    items = []
    for i in range(num_items):
        width = random.uniform(min_dim, max_dim)
        height = random.uniform(min_dim, max_dim)
        depth = random.uniform(min_dim, max_dim)
        items.append(Item(id=i+1, width=width, height=height, depth=depth))
    return items

def draw_box(ax, x, y, z, width, height, depth, color='blue', alpha=0.6, is_container=False):
    """Draw a 3D box"""
    if is_container:
        # Draw container as wireframe only
        # Bottom face
        ax.plot([x, x+width], [y, y], [z, z], 'k-', linewidth=2)
        ax.plot([x+width, x+width], [y, y+height], [z, z], 'k-', linewidth=2)
        ax.plot([x+width, x], [y+height, y+height], [z, z], 'k-', linewidth=2)
        ax.plot([x, x], [y+height, y], [z, z], 'k-', linewidth=2)
        
        # Top face
        ax.plot([x, x+width], [y, y], [z+depth, z+depth], 'k-', linewidth=2)
        ax.plot([x+width, x+width], [y, y+height], [z+depth, z+depth], 'k-', linewidth=2)
        ax.plot([x+width, x], [y+height, y+height], [z+depth, z+depth], 'k-', linewidth=2)
        ax.plot([x, x], [y+height, y], [z+depth, z+depth], 'k-', linewidth=2)
        
        # Vertical edges
        ax.plot([x, x], [y, y], [z, z+depth], 'k-', linewidth=2)
        ax.plot([x+width, x+width], [y, y], [z, z+depth], 'k-', linewidth=2)
        ax.plot([x+width, x+width], [y+height, y+height], [z, z+depth], 'k-', linewidth=2)
        ax.plot([x, x], [y+height, y+height], [z, z+depth], 'k-', linewidth=2)
    else:
        # Draw filled box for items
        # Define vertices
        vertices = np.array([
            [x, y, z],                      # 0
            [x + width, y, z],              # 1
            [x + width, y + height, z],     # 2
            [x, y + height, z],             # 3
            [x, y, z + depth],              # 4
            [x + width, y, z + depth],      # 5
            [x + width, y + height, z + depth], # 6
            [x, y + height, z + depth]      # 7
        ])
        
        # Define faces
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom (z=z)
            [vertices[4], vertices[7], vertices[6], vertices[5]],  # top (z=z+depth)
            [vertices[0], vertices[4], vertices[5], vertices[1]],  # front (y=y)
            [vertices[2], vertices[6], vertices[7], vertices[3]],  # back (y=y+height)
            [vertices[1], vertices[5], vertices[6], vertices[2]],  # right (x=x+width)
            [vertices[4], vertices[0], vertices[3], vertices[7]]   # left (x=x)
        ]
        
        # Add faces
        for face in faces:
            poly = [[face[i] for i in range(len(face))]]
            ax.add_collection3d(Poly3DCollection(poly, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5))

def visualize_packing(container: Container, packed_items: List[Item], unpacked_items: List[Item]):
    """Visualize the 3D bin packing result"""
    fig = plt.figure(figsize=(15, 10))
    
    # Create main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Draw container outline with wireframe only
    draw_box(ax1, 0, 0, 0, container.width, container.height, container.depth, 
             color='gray', alpha=0.1, is_container=True)
    
    # Draw packed items with different colors
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(packed_items), 1)))
    
    for i, item in enumerate(packed_items):
        color = colors[i] if len(packed_items) > 0 else 'blue'
        draw_box(ax1, item.x, item.y, item.z, item.width, item.height, item.depth,
                color=color, alpha=0.7, is_container=False)
        
        # Add item ID label at center
        center_x = item.x + item.width / 2
        center_y = item.y + item.height / 2
        center_z = item.z + item.depth / 2
        ax1.text(center_x, center_y, center_z, str(item.id), fontsize=8, ha='center', va='center')
    
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    ax1.set_zlabel('Depth')
    ax1.set_title(f'3D Bin Packing Result\nPacked: {len(packed_items)}, Unpacked: {len(unpacked_items)}')
    
    # Set equal aspect ratio and limits
    max_dim = max(container.width, container.height, container.depth)
    ax1.set_xlim([0, container.width])
    ax1.set_ylim([0, container.height])
    ax1.set_zlim([0, container.depth])
    
    # Set aspect ratio to be equal
    ax1.set_box_aspect([container.width, container.height, container.depth])
    '''
    # Create statistics plot
    ax2 = fig.add_subplot(222)
    
    # Utilization pie chart
    used_vol = container.used_volume
    unused_vol = container.volume - used_vol
    
    ax2.pie([used_vol, unused_vol], 
            labels=['Used Space', 'Free Space'], 
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightblue'])
    ax2.set_title(f'Container Utilization\nTotal Volume: {container.volume:.1f}')
    
    # Items size distribution
    ax3 = fig.add_subplot(223)
    
    if packed_items:
        packed_volumes = [item.volume for item in packed_items]
        ax3.hist(packed_volumes, bins=min(10, len(packed_items)), alpha=0.7, label='Packed Items', color='green')
    
    if unpacked_items:
        unpacked_volumes = [item.volume for item in unpacked_items]
        ax3.hist(unpacked_volumes, bins=min(10, len(unpacked_items)), alpha=0.7, label='Unpacked Items', color='red')
    
    ax3.set_xlabel('Item Volume')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Item Volume Distribution')
    if packed_items or unpacked_items:
        ax3.legend()
    
    # Summary statistics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    stats_text = f"""
    PACKING STATISTICS
    ==================
    Container Dimensions: {container.width:.1f} x {container.height:.1f} x {container.depth:.1f}
    Container Volume: {container.volume:.1f}
    
    Items Packed: {len(packed_items)}
    Items Unpacked: {len(unpacked_items)}
    Packing Success Rate: {len(packed_items)/(len(packed_items)+len(unpacked_items))*100:.1f}%
    
    Volume Used: {used_vol:.1f}
    Volume Unused: {unused_vol:.1f}
    Space Utilization: {container.utilization:.1f}%
    """
    
    if packed_items:
        packed_volumes = [item.volume for item in packed_items]
        stats_text += f"\nAverage Packed Item Volume: {np.mean(packed_volumes):.1f}"
    
    if unpacked_items:
        unpacked_volumes = [item.volume for item in unpacked_items]
        stats_text += f"\nAverage Unpacked Item Volume: {np.mean(unpacked_volumes):.1f}"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    '''
    plt.tight_layout()
    plt.show()

def create_items_dataframe(packed_items: List[Item], unpacked_items: List[Item]) -> pd.DataFrame:
    """Create a summary DataFrame of all items"""
    all_items = packed_items + unpacked_items
    
    data = []
    for item in all_items:
        data.append({
            'Item_ID': item.id,
            'Width': round(item.width, 2),
            'Height': round(item.height, 2),
            'Depth': round(item.depth, 2),
            'Volume': round(item.volume, 2),
            'Packed': item.packed,
            'Position_X': round(item.x, 2) if item.packed else None,
            'Position_Y': round(item.y, 2) if item.packed else None,
            'Position_Z': round(item.z, 2) if item.packed else None,
        })
    
    return pd.DataFrame(data)

# ===== MAIN EXECUTION ROUTINE =====

def main_bin_packing_routine():
    """Step-by-step routine for 3D bin packing visualization"""
    
    print("🏭 3D Bin Packing for Warehouse Management")
    print("=" * 50)
    
    # Step 1: Define container dimensions
    print("\n📦 Step 1: Defining Container")
    container_width = 20
    container_height = 15
    container_depth = 25
    container = Container(container_width, container_height, container_depth)
    print(f"Container dimensions: {container_width} x {container_height} x {container_depth}")
    print(f"Container volume: {container.volume}")
    
    # Step 2: Generate random items
    print("\n📋 Step 2: Generating Random Items")
    num_items = 15
    items = generate_random_items(num_items, min_dim=2, max_dim=8)
    print(f"Generated {num_items} items with random dimensions")
    
    total_items_volume = sum(item.volume for item in items)
    print(f"Total items volume: {total_items_volume:.1f}")
    print(f"Theoretical max utilization: {(total_items_volume/container.volume)*100:.1f}%")
    
    # Step 3: Initialize packer and pack items
    print("\n🎯 Step 3: Packing Items")
    packer = BinPacker3D(container)
    packed_items, unpacked_items = packer.pack_items(items)
    
    print(f"Successfully packed: {len(packed_items)} items")
    print(f"Failed to pack: {len(unpacked_items)} items")
    print(f"Actual space utilization: {container.utilization:.1f}%")
    
    # Step 4: Create summary DataFrame
    print("\n📊 Step 4: Creating Summary Report")
    df = create_items_dataframe(packed_items, unpacked_items)
    print("\nItems Summary:")
    print(df.head(10))
    
    # Step 5: Visualize results
    print("\n📈 Step 5: Visualizing Results")
    visualize_packing(container, packed_items, unpacked_items)
    
    # Step 6: Additional analysis
    print("\n🔍 Step 6: Additional Analysis")
    if unpacked_items:
        print("\nUnpacked Items Analysis:")
        unpacked_df = df[df['Packed'] == False]
        print(f"Largest unpacked item volume: {unpacked_df['Volume'].max():.1f}")
        print(f"Average unpacked item volume: {unpacked_df['Volume'].mean():.1f}")
    
    packed_df = df[df['Packed'] == True]
    if len(packed_df) > 0:
        print(f"\nPacked Items Analysis:")
        print(f"Largest packed item volume: {packed_df['Volume'].max():.1f}")
        print(f"Average packed item volume: {packed_df['Volume'].mean():.1f}")
    
    return container, packed_items, unpacked_items, df

# Run the complete routine
if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Execute the main routine
    container, packed_items, unpacked_items, summary_df = main_bin_packing_routine()
    
    print(f"\n✅ Bin packing completed!")
    print(f"Final utilization: {container.utilization:.1f}%")
