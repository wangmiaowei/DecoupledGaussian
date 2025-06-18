import time
import numpy as np
import viser

# Initialize the VisorServer instance
server = viser.ViserServer(port=8085)

def draw_rectangle_bbox(center, length, width, height, color, bbox_id):
    # Define the offsets of the 8 vertices of the rectangle relative to the center point
    corner_offsets = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [-0.5, 0.5,  0.5], [0.5, 0.5,  0.5]
    ])

    # Calculate the actual position of the vertices
    scales = np.array([length, width, height])
    positions = corner_offsets * scales + center
    
    # Define the 12 sides of a rectangle
    edges = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])

    # Draw each side of the rectangle
    for i, edge in enumerate(edges):
        server.scene.add_spline_catmull_rom(
            f"/bbox_{bbox_id}_edge_{i}",
            positions[edge],
            tension=0.8,
            line_width=1.0,
            color=color,
            segments=100,
        )

# Example call
color = np.array([1.0, 0, 0])
draw_rectangle_bbox(center=[0, 0, 0], length=2, width=1, height=1.5, color=color, bbox_id=0)

# Start the server
while True:
    time.sleep(1)