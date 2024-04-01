from fastapi import FastAPI, UploadFile, File
import shutil
import os
import subprocess

app = FastAPI()

def run_detection(source_image_path, weights_path):
    command = ["python", "yolo", "detect", "predict", f"model=runs/detect/train7/weights/best.pt", "conf=0.25", f"source={source_image_path}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return_value = stdout.decode().strip()
    
    if process.returncode != 0:
        return f"Error occurred: {stderr.decode('utf-8')}"
    else:
        return f"Detection successful:\n{stdout.decode('utf-8')}"

@app.post("/predict/")
async def predict(source_image: UploadFile = File(...)):
    # Save the uploaded image
    image_save_path = "temp_image.jpg"
    with open(image_save_path, "wb") as buffer:
        shutil.copyfileobj(source_image.file, buffer)
    
    # Define weights path
    weights_path = "runs/detect/train7/weights/best.pt"

    # Run detection
    detection_result = run_detection(image_save_path, weights_path)
    
    # Clean up temporary files
    os.remove(image_save_path)
    
    return {"result": detection_result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
