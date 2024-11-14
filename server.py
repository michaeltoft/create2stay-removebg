import os
import io
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from rembg import remove, new_session
from PIL import Image
import requests

app = FastAPI()

# Create a session at startup
session = new_session("u2net")

async def download_image(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=404, detail="Image not found")
            return await response.read()

@app.get("/removebg")
async def remove_background(url: str):
    try:
        # Download the image
        image_data = await download_image(url)

        # Convert to PIL Image and maintain orientation
        input_image = Image.open(io.BytesIO(image_data))

        # Process the image with rembg
        output_image = remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=255,
            alpha_matting_background_threshold=0,
            alpha_matting_erode_size=5,
            post_process_mask=True
        )

        # Save to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG', optimize=False, quality=100)
        img_byte_arr.seek(0)

        # Return the processed image
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)