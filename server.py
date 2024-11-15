import os
import io
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from rembg import remove, new_session
from PIL import Image

app = FastAPI()

# Create a session at startup
session = new_session("u2net")

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
        padding: int = Query(0, ge=0, description="Padding to apply around the image")
):
    try:
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