from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
import io
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import google.generativeai as genai
import shutil
import os
import subprocess


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def dynamic_file(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/PatientForm")
async def Patient_form(request: Request):
    return templates.TemplateResponse("PatientForm.html", {"request": request})

@app.get("/report")
async def report_fun(request: Request):
    return templates.TemplateResponse("report.html", {"request": request})


@app.post("/upload")
async def report(request: Request, file: UploadFile = File(...)):
    data = await file.read()

    # Convert the bytes data to a NumPy array
    nparr = np.frombuffer(data, np.uint8)
    # Decode the image using cv2.imdecode
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (28, 28))
    
    model = tf.keras.models.load_model("./best_model.h5")
    result = model.predict(img_resized.reshape(1, 28, 28, 3))

    max_prob = max(result[0])
    classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}

    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]
    print(class_name)
    _, img_encoded = cv2.imencode('.png', img_resized)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    result = {
        "img": img_base64,
        "prediction": class_name
    }
    return templates.TemplateResponse("PatientForm.html", {"request": request,  "img": img_base64, "result":class_name })

@app.get("/chat", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get_gemini_completion", response_class=HTMLResponse)
async def get_gemini_completion(
    request: Request,
    gemini_api_key: str = Form(...),
    prompt: str = Form(...),
):
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=['space'],
                max_output_tokens=400,
                temperature=0)
        )
        print(response.text)
        return templates.TemplateResponse("chat.html", {"request": request, "response": response.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def run_detection(source_image_path, weights_path):
    command = f'!yolo task=./runs/detect mode=predict model=./runs/detect/train7/weights/best.pt conf=0.25 source="{source_image_path}"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return_value = stdout.decode().strip()
    
    if process.returncode != 0:
        return f"Error occurred: {stderr.decode('utf-8')}"
    else:
        return f"Detection successful:\n{stdout.decode('utf-8')}"

UPLOAD_FOLDER = 'static'
@app.post("/predict/", response_class=HTMLResponse)
async def predict(source_image: UploadFile = File(...)):
    # Save the uploaded image
    print("hello")
    image_path = f"{UPLOAD_FOLDER}/{source_image.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, source_image.filename)

    with open(save_path, "wb") as image:
         content = await source_image.read()
         image.write(content)
    
    # Define weights path
    weights_path = "./runs/detect/train7/weights/best.pt"
    path = "./static/mel1.jpeg"
    # Run detection
    detection_result = run_detection(path, weights_path)
    
    # Clean up temporary files
    # os.remove(save_path)
    
    return {"result": detection_result}

# @app.post("/upload_image", response_class=HTMLResponse)
# async def upload_image( request: Request,image_file: UploadFile = File(...)):
#     image_path = f"{UPLOAD_FOLDER}/{image_file.filename}"
#     save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    
#     with open(save_path, "wb") as image:
#         content = await image_file.read()
#         image.write(content)
    
#     predictions = process_image(image_path)
    
#     context = {
#         "request": request,
#
