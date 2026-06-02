#!/usr/bin/env python3
# combine_graphs.py
# Usage: python3 combine_graphs.py

from PIL import Image
import sys

def main():
    # 1. Define the input image files in the correct order (Left to Right)
    image_files = ['graph_a.png', 'graph_b.png', 'graph_c.png']
    images = []

    # 2. Load images
    try:
        for file in image_files:
            img = Image.open(file)
            images.append(img)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find the image file: {e.filename}")
        print("Make sure you generated the .png files first!")
        sys.exit(1)

    # 3. Calculate total width and max height for the canvas
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # 4. Create a new blank white canvas
    combined_im = Image.new('RGB', (total_width, max_height), color='white')

    # 5. Paste the images side-by-side
    x_offset = 0
    for im in images:
        # If an image is shorter than max_height, center it vertically
        y_offset = (max_height - im.size[1]) // 2 
        combined_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]

    # 6. Save the final combined image
    output_filename = 'combined_graphs_ABC.png'
    combined_im.save(output_filename)
    print(f"\n[SYSTEM] Successfully stitched {len(images)} images horizontally!")
    print(f"[SYSTEM] Output saved to: {output_filename}")

if __name__ == "__main__":
    # Ensure Pillow is installed
    try:
        import PIL
    except ImportError:
        print("[ERROR] Pillow library is not installed. Run: pip install Pillow")
        sys.exit(1)
        
    main()