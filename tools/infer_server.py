import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import scatter

from mmgs.utils import auto_select_device
from mmgs.apis.inference import init_model, inference_model, update_force, update_cam, update_simulation
import plotly.express as px

# server.py
import io
import base64
from PIL import Image
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


def parse_args():
    parser = argparse.ArgumentParser(description='mmgd test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()

    # assert args.metrics or args.out, \
    #     'Please specify at least one of output path and evaluation metrics.'

    return args

class InteractiveSimulator:
    
    def __init__(self, args):
        self.args = args

        # Initialization
        self.model = init_model(args.config, args.checkpoint, device=args.device or auto_select_device(), force_forward=-1)
        self.data, self.active_mask, self.device = inference_model(self.model)
        self.cur_state = None
        self.prev_state = None
        self.pred_frame_idx = 1

        self.data = scatter(self.data, [self.device])[0]
        self.active_mask = scatter(self.active_mask, [self.device])[0]


def render_gaussian_splatting(img_array):
    """
    1) Use your existing code to produce a PIL image or NumPy array.
    2) Convert that to a base64 string that the client can set as <img src="data:image/...">
    """
    # For example, suppose you have a function `gaussian_splatter(simulation_state)` 
    # that returns a PIL Image or a NumPy array. 
    # We'll mock it with a blank image for demonstration:

    img = Image.fromarray(img_array)
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # We'll send the data URL
    data_url = f"data:image/jpeg;base64,{encoded_img}"
    return data_url


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect_handler():
    print("Client connected")
    # Send initial rendered frame
    # frame_data = render_gaussian_splatting()
    # emit('renderFrame', frame_data)

@socketio.on('applyForce')
def handle_apply_force(data):
    # print("Applying force")
    raw_force = data['force']
    force = torch.from_numpy(np.array([raw_force['x'], raw_force['y'], raw_force['z']], dtype=np.float32).reshape(1, 3))
    force = scatter(force, [handler.device])[0]
    handler.cur_state = update_force(handler.data['inputs']['pin_mask'], handler.active_mask, force, cur_state=handler.cur_state, dt=1/30)

@socketio.on('updateCamera')
def handle_update_camera(data):
    print(f"Update camera: {data}")
    cur_cam = handler.data['inputs']['cam'][0]
    new_cam = update_cam(cur_cam, data['rotation'], data['translation'])
    new_cam.to_device(next(handler.model.parameters()).device)
    handler.data['inputs']['cam'] = [new_cam]


def simulation_loop():
    # Update
    while True:
        # socketio.sleep(0.016)  # ~ 60 FPS
        socketio.sleep(1/10)
        ## Update forces
        # cur_state = update_force(active_mask, force, cur_state=cur_state, dt=1/30)
        # data = update_cam(data, cam)
        img, prev_state, cur_state, pred_frame_idx = update_simulation(handler.model, handler.data, handler.prev_state, handler.cur_state, cur_cov=None, pred_frame_idx=handler.pred_frame_idx, zero_init=False)
        # Rollout
        handler.pred_frame_idx = min(pred_frame_idx, 2) # Just for convinence, no need to update gt_label
        handler.cur_state = scatter(cur_state, [handler.device])[0]
        handler.prev_state = scatter(prev_state, [handler.device])[0]
        # Output images
        img_url = render_gaussian_splatting(img)
        # Broadcast the new frame to all clients
        socketio.emit('renderFrame', img_url)


# if __name__ == '__main__':
#     main()

args = parse_args()
handler = InteractiveSimulator(args)

if __name__ == '__main__':
    socketio.start_background_task(simulation_loop)
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)