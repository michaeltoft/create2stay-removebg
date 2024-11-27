import os
import io
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import re

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rest of your code remains the same...
session = new_session("u2net")

def is_valid_hex_color(color: str) -> bool:
    """Validate hex color format."""
    if not color:
        return False
    # Check for 3 or 6 character hex color with optional #
    pattern = r'^#?([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
    return bool(re.match(pattern, color))

def hex_to_rgba(hex_color: str) -> tuple:
    """Convert hex color to RGBA tuple."""
    # Remove # if present
    hex_color = hex_color.lstrip('#')

    # Convert 3-digit hex to 6-digit
    if len(hex_color) == 3:
        hex_color = ''.join(c * 2 for c in hex_color)

    # Convert to RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b, 255)  # Full opacity

def apply_background_color(image: Image.Image, color: str) -> Image.Image:
    """Apply background color to transparent image."""
    # Create a new image with the solid background
    background = Image.new('RGBA', image.size, hex_to_rgba(color))

    # Paste the original image using itself as mask
    background.paste(image, mask=image)

    return background

async def download_image(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=404, detail="Image not found")
            return await response.read()

def trim_image(image: Image.Image) -> Image.Image:
    """Trim transparent pixels from image edges."""
    # Get the alpha channel
    alpha = image.getchannel('A')
    # Get the bounding box of non-zero regions
    bbox = alpha.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

def resize_with_padding(image: Image.Image, width: int, height: int, padding: int) -> Image.Image:
    """Resize image to target dimensions with padding."""
    # Calculate target size without padding
    target_width = width - (2 * padding)
    target_height = height - (2 * padding)

    # Get current image size
    img_width, img_height = image.size

    # Calculate aspect ratios
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height

    # Calculate new dimensions maintaining aspect ratio
    if img_ratio > target_ratio:
        # Width is the limiting factor
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        # Height is the limiting factor
        new_height = target_height
        new_width = int(target_height * img_ratio)

    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with padding
    padded_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Calculate position to paste resized image
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2

    # Paste resized image onto padded canvas
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image

@app.get("/removebg")
async def remove_background(
        url: str,
        width: int = Query(None, gt=0, description="Target width including padding"),
        height: int = Query(None, gt=0, description="Target height including padding"),
        padding: int = Query(0, ge=0, description="Padding to apply around the image"),
        bgcolor: str = Query(None, description="Hex color for background (e.g., '#FF0000' or 'FF0000')")
):
    try:
        # Validate bgcolor if provided
        if bgcolor and not is_valid_hex_color(bgcolor):
            raise HTTPException(
                status_code=400,
                detail="Invalid hex color format. Use '#RRGGBB', 'RRGGBB', '#RGB', or 'RGB'"
            )

        # Download the image
        image_data = await download_image(url)

        # Convert to PIL Image
        input_image = Image.open(io.BytesIO(image_data))

        # Convert to RGBA if not already
        if input_image.mode != 'RGBA':
            input_image = input_image.convert('RGBA')

        # Process the image with rembg
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=250,
            alpha_matting_background_threshold=5,
            alpha_matting_erode_size=1,
            post_process_mask=True
        )

        # Apply trimming, resizing and padding if dimensions are provided
        if width is not None and height is not None:
            # Trim transparent edges
            output_image = trim_image(output_image)
            # Resize and add padding
            output_image = resize_with_padding(output_image, width, height, padding)

        # Apply background color if provided
        if bgcolor:
            output_image = apply_background_color(output_image, bgcolor)

        # Save to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG', optimize=False, quality=100)
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)