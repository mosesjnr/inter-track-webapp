# 1. Library imports
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware


class Rent(BaseModel):
    rooms: int
    location: str 


# 2. Create app and model objects
app = FastAPI()
model = joblib.load("joblib_model.sav")

# 3. Expose the prediction functionality, make a prediction from the passed

@app.get("/")
def read_root():
    return {"Hello": "Stutern"}

@app.post("/predict/")
def predict_rent(data: Rent):
    data = data.dict()
    rooms = data['rooms']
    location = data['location']
    result = np.exp(model.predict([rooms, location]))
    result = np.round(result, 2)
    
    return {'Expected rent is': result}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
