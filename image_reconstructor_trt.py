import torch
import cv2
import numpy as np
from model.model import *
from utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, ImageWriter, UnsharpMaskFilter
from utils.inference_utils import upsample_color_image, merge_channels_into_color_image  # for color reconstruction
from utils.util import robust_min, robust_max
from utils.timers import CudaTimer, cuda_timers
from os.path import join
from collections import deque
import torch.nn.functional as F
from torch2trt import torch2trt



class ImageReconstructor:
    def __init__(self, model,height, width, num_bins, options):

        self.model = model
        self.use_gpu = options.use_gpu
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.height = height # 240 
        self.width = width # 320
        self.num_bins = num_bins # 5

        self.initialize(self.height, self.width, options)

    def initialize(self, height, width, options):
        print('== Image reconstruction == ')
        print('Image size: {}x{}'.format(self.height, self.width))

        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        self.perform_color_reconstruction = options.color  # whether to perform color reconstruction (only use this with the DAVIS346color)
        if self.perform_color_reconstruction:
            if options.auto_hdr:
                print('!!Warning: disabling auto HDR for color reconstruction!!')
            options.auto_hdr = False  # disable auto_hdr for color reconstruction (otherwise, each channel will be normalized independently)

        self.crop = CropParameters(self.width, self.height, 4) # num_encoders = 4

        init_1 = torch.zeros((1,16,240,320),device='cuda:0',dtype=torch.float16)
        init_2 = torch.zeros((1,16,240,320),device='cuda:0',dtype=torch.float16)
        init_state = [init_1,init_2]
        self.last_states_for_each_channel = {'grayscale': init_state}

        if self.perform_color_reconstruction:
            self.crop_halfres = CropParameters(int(width / 2), int(height / 2),
                                               4)
            for channel in ['R', 'G', 'B', 'W']:
                self.last_states_for_each_channel[channel] = None

        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.image_writer = ImageWriter(options)
        self.image_display = ImageDisplay(options)

    def update_reconstruction(self, event_tensor, event_tensor_id, stamp=None):
        with torch.no_grad():

            with CudaTimer('Reconstruction'):

                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    events = event_tensor.unsqueeze(dim=0) # from (5,240,320) to (1,5,240,320)
                    events = events.to(self.device)

                events = self.event_preprocessor(events)
                # print(events.max(),events.min())
                # events = events.type(torch.float16)

                # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
                events_for_each_channel = {'grayscale': self.crop.pad(events)}
                
                reconstructions_for_each_channel = {}
                if self.perform_color_reconstruction:
                    events_for_each_channel['R'] = self.crop_halfres.pad(events[:, :, 0::2, 0::2])
                    events_for_each_channel['G'] = self.crop_halfres.pad(events[:, :, 0::2, 1::2])
                    events_for_each_channel['W'] = self.crop_halfres.pad(events[:, :, 1::2, 0::2])
                    events_for_each_channel['B'] = self.crop_halfres.pad(events[:, :, 1::2, 1::2])

                # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
                for channel in events_for_each_channel.keys():
                    with CudaTimer('Inference'):
                        new_predicted_frame,state1,state2 = self.model(events_for_each_channel[channel],self.last_states_for_each_channel[channel][0],self.last_states_for_each_channel[channel][1])

                    new_predicted_frame = new_predicted_frame.squeeze(dim=0)
                    state1 = state1.squeeze(dim=0)
                    state2 = state2.squeeze(dim=0)
                    if self.no_recurrent:
                        self.last_states_for_each_channel[channel] = None
                    else:
                        self.last_states_for_each_channel[channel] = [state1,state2]
    
                    # Output reconstructed image
                    crop = self.crop if channel == 'grayscale' else self.crop_halfres
                    # print(crop.ix0,crop.ix1,crop.iy0,crop.iy1) ## 0, 320, 0, 240
                    # print(new_predicted_frame.shape) ## ([1,1,240,320])

                    # Unsharp mask (on GPU)
                    # new_predicted_frame = new_predicted_frame.type(torch.float32)
                    new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                    # Intensity rescaler (on GPU)
                    new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                    with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                        reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                        crop.ix0:crop.ix1].cpu().numpy()

                if self.perform_color_reconstruction:
                    out = merge_channels_into_color_image(reconstructions_for_each_channel)
                else:
                    out = reconstructions_for_each_channel['grayscale']

            # Post-processing, e.g bilateral filter (on CPU)
            out = self.image_filter(out)

            self.image_writer(out, event_tensor_id, stamp, events=events)
            self.image_display(out, events)
