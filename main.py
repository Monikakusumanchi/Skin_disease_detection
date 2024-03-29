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