# 1. Library imports

from numpy import double
from pydantic import BaseModel 
import uvicorn
from fastapi import FastAPI
import yolo



# 2. Create the app object
from absl import app, flags, logging
from absl.flags import FLAGS

'''
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')

#flags.DEFINE_list('images', imagenes, 'path to input image')


flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
'''

model = yolo

app = FastAPI()

class Data(BaseModel):
    path: str #image path
    dimension: tuple #dimension de la imagen



# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    '''
    This is a first docstring.
    '''
    return {'message': 'Hello, strange'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.post('/yolo_v4/')
def detect_text(params: Data):
    '''
    Obtiene las cordenas del texto en una imagen.
    Como parametros recibe la imagen, la dimencion y si se desea mostrar o no el texto
    '''
    

    #model.yolo_v4
    x_min,y_min,x_max,y_max = model.yolo_v4#(params.path)

    '''
    return {
             'x_min': x_min,
             'y_min': y_min,
             'x_max':x_max,
             'y_max': y_max,
          }
'''


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8002)   