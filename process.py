import os
from rembg import remove
from PIL import Image

def process_images():
    input_dir = '/app/input'
    output_dir = '/app/output'

    # Process all images in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            # Input and output paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'processed_{filename[:-4]}.png')

            try:
                # Read image
                input_image = Image.open(input_path)

                # Remove background
                output_image = remove(input_image)

                # Save the result
                output_image.save(output_path)
                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    print("Starting background removal process...")
    process_images()
    print("Finished processing all images.")