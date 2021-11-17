# coding=utf-8
import os
from pathlib import Path
from queue import Queue
import argparse
from time import monotonic
from annoy import AnnoyIndex
import cv2
import depthai
import numpy as np
from imutils.video import FPS
import h5py
import time
import warnings
import sys
import torch
import torchvision.transforms as transforms
sys.path.append(".")
import numpy as np
import math
from PIL import Image
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.data_io import transform as trans

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="prevent debug output"
)
parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)",
)

parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="The path of the video file used for inference (conflicts with -cam)",
)
#parser.add_argument("--model_dir",type=str,default=os.path.normpath(str(os.getcwd())+"/models/anti_spoof_models"),help="model_lib used to test")
parser.add_argument(
    "--pro_t",
    required=True,
    help="probability threshold for face identification",
)
parser.add_argument(
    "--corr_t",
    required=True,
    help="correlation threshold for face identification",
)

args = parser.parse_args()

debug = not args.no_debug

if args.camera and args.video:
    raise ValueError(
        'Command line parameter error! "-Cam" cannot be used together with "-vid"!'
    )
elif args.camera is False and args.video is None:
    raise ValueError(
        'Missing inference source! Use "-cam" to run on DepthAI cameras, or use "-vid <path>" to run on video files'
    )


def to_planar(arr: np.ndarray, shape: tuple):
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()
def to_planar1(arr: torch.tensor):
    return arr
def to_planar2(arr: np.ndarray, shape: tuple):
    arr=cv2.resize(arr, shape).transpose((2, 0, 1))
    arr=torch.from_numpy(arr)
    arr = arr.unsqueeze(0).to(torch.device("cpu"))
    
    return arr
def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())
def to_nn_result1(nn_data):
    print(type(nn_data.getFirstLayerFp16()))
    return nn_data.getFirstLayerFp16()
def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    return x_out.tryGet()

def frame_norm(frame, *xy_vals):
    return (
        np.clip(np.array(xy_vals), 0, 1) * np.array(frame * (len(xy_vals) // 2))[::-1]
    ).astype(int)
def correction(frame, angle=None, invert=False):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1)
    affine = cv2.invertAffineTransform(mat).astype("float32")
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if invert:
        return corr, affine
    return corr
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
def softmax(x):
    
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
class DepthAI:
    def __init__(
        self,
        file=None,
        camera=False,

    ):
        print("Loading pipeline...")

        self.file = file
        self.camera = camera
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 0.45 if self.camera else 2
        self.lineType = 0 if self.camera else 3

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()
        
        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            self.cam = self.pipeline.createColorCamera()
            self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
            self.cam.setResolution(
                depthai.ColorCameraProperties.SensorResolution.THE_4_K
            )
            self.cam.setInterleaved(False)
            self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

            self.cam_xout = self.pipeline.createXLinkOut()
            self.cam_xout.setStreamName("preview")
            self.cam.preview.link(self.cam_xout.input)

        self.create_nns()
        
        print("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path: str, model_name: str, first: bool = False):
        """

        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first and self.camera:
            print("linked cam.preview to model_nn.input")
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def create_mobilenet_nn(
        self,
        model_path: str,
        model_name: str,
        conf: float = 0.5,
        first: bool = False,
    ):
        """

        :param model_path: model name
        :param model_name: model abbreviation
        :param conf: confidence threshold
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createMobileNetDetectionNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)

        if first and self.camera:
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")
        self.device.startPipeline()

        self.start_nns()

        if self.camera:
            self.preview = self.device.getOutputQueue(
                name="preview", maxSize=4, blocking=False
            )

    def start_nns(self):
        pass

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )

    def draw_bbox(self, bbox, color):
        cv2.rectangle(
            img=self.debug_frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=2,
        )

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        s = self.parse_fun()
        # if s :
        #     raise StopIteration()
        if debug:
            self.debug_frame=cv2.resize(self.debug_frame,(550,550))
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                print(
                    f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
                )
                raise StopIteration()

    def parse_fun(self):
        pass

    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break

            try:
                self.parse()
            except StopIteration:
                break

        cap.release()

    def run_camera(self):
        
        while True:
            in_rgb = self.preview.tryGet()
            
            #if in_rgb is not None and frame_counter %1==0:
            if in_rgb is not None:
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                self.frame = (
                    in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                )
                self.frame = np.ascontiguousarray(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break

    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v

    def run(self):
        self.fps_cam.start()
        self.fps_nn.start()
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device

        
    def face_recognize(self,vector):

        h5f = h5py.File('data/data_vector_image_all.h5','r')
        data_vec = h5f['dataset'][:]
        h5f.close()
        r = h5py.File('data/data_account_image_all.h5','r')
        tree= AnnoyIndex(128, 'angular')  
        tree.load('data/data_vector_image_all.ann')     
        data_account = r['sentences'][:]
        r.close()  
        check=np.array(tree.get_nns_by_vector(vector, len(data_account),include_distances=True))  
        i=np.argmin(check[1,:])  
        return data_account[:,:1][int(check[0,i])][0],check[1,i],data_vec[int(check[0,i]),:]

class Main(DepthAI):
    def __init__(self, file=None, camera=False, pro_threshold=None, corr_threshold=None):
        self.cam_size = (300,300)
        super(Main, self).__init__(file, camera)
        self.face_pro = Queue()
        self.face_frame_reg=Queue()
        self.face_frame = Queue()
        self.face_coords = Queue()
        self.face_frame_corr=Queue()
        self.face_coord=Queue()
        self.image_cropper = CropImage()
        self.pro_threshold=pro_threshold
        self.corr_threshold=corr_threshold

    def fas(self,image,image_bbox,model_name):

        h_input, w_input, _, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        #print(param)
        if scale is None:
            param["crop"] = False
        img = self.image_cropper.crop(**param) 
        return img

    def trans(self,img):
        
        img = Image.fromarray(img)
        img_size=224
        ratio = 224.0 / float(img_size)
        # Data loading code
        normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956],  ##accorcoding to casia-surf val to commpute
                                        std=[0.10050353, 0.100842826, 0.10034215])
        transform = transforms.Compose([
            transforms.Resize(int(256 * ratio)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
        image = transform(img).unsqueeze(0)
        return image

    def create_nns(self):
        # self.create_mobilenet_nn(
        #     "models/face-detection-retail-0004_openvino_2021.2_4shave.blob",
        #     "mfd",
        #     first=self.camera,
        # )
        self.create_nn(
            "models/feathernetB.blob",
            "anti_spoofing3"
        )
        # self.create_nn(
        #     "models/4_0_0_80x80_MiniFASNetV1SE.blob",
        #     "anti_spoofing1"
        # )
        # self.create_nn(
        #     "models/2_7_80x80_MiniFASNetV2.blob",
        #     "anti_spoofing2"
        # )
        self.create_mobilenet_nn(
            "models/face-mask-detection-mobilenet-ssdv2_openvino_2021.2_4shave.blob",
            "mask",
            first=self.camera,
        )
        self.create_nn(
            "models/head-pose-estimation-adas-0001_openvino_2021.2_4shave.blob",
            "head_pose",
        )
        self.create_nn("models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob",'insightface')

    def start_nns(self):

        if not self.camera:
            self.mfd_in = self.device.getInputQueue("mfd_in")
            self.mask_in = self.device.getInputQueue("mask_in",4,False)
        self.mfd_nn = self.device.getOutputQueue("mfd_nn", 4, False)
        self.mask_nn = self.device.getOutputQueue("mask_nn",4,False)
        # self.anti_spoofing1_in = self.device.getInputQueue("anti_spoofing1_in", 4, False)
        # self.anti_spoofing1_nn = self.device.getOutputQueue("anti_spoofing1_nn", 4, False)
        # self.anti_spoofing2_in = self.device.getInputQueue("anti_spoofing2_in", 4, False)
        # self.anti_spoofing2_nn = self.device.getOutputQueue("anti_spoofing2_nn", 4, False)
        self.anti_spoofing3_in = self.device.getInputQueue("anti_spoofing3_in", 4, False)
        self.anti_spoofing3_nn = self.device.getOutputQueue("anti_spoofing3_nn", 4, False)
        self.head_pose_in = self.device.getInputQueue("head_pose_in", 4, False)
        self.head_pose_nn = self.device.getOutputQueue("head_pose_nn", 4, False)
        self.insightface_in = self.device.getInputQueue("insightface_in", 4, False)
        self.insightface_nn = self.device.getOutputQueue("insightface_nn", 4, False)
        
    def run_face_mn(self):
        '''
        # Region only uses face detection

        nn_data=self.mfd_nn.tryGet()
        if nn_data is None :
            return False
        bboxes = nn_data.detections
        for bbox in bboxes:  
            face_coord = frame_norm(
                self.frame.shape[:2], *[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            )
            self.face_frame_reg.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )
            self.face_frame.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )
            self.face_coords.put(face_coord)
            
            self.draw_bbox(face_coord, (10, 245, 10))
        '''
        # Region uses both face datection and mask detection
        labelMap1 = ["background", "no mask", "mask", "no mask"]
        nn_data1 = self.mask_nn.tryGet()
        if nn_data1 is None :
            return False
        mask_det=nn_data1.detections
        for detection in mask_det:
            face_coord = frame_norm(
                self.frame.shape[:2], *[detection.xmin, detection.ymin, detection.xmax, detection.ymax]
            )
            self.face_frame.put(
                self.frame
            )
            self.face_frame_reg.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )
            self.face_frame.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )
            self.face_coords.put(face_coord)
            self.face_coord.put(face_coord)
            self.draw_bbox(face_coord, (10, 245, 10))
            self.put_text(
                labelMap1[detection.label],
                (215,10),(0,0,255) ,0.45,
            )
        return True

    def run_anti_spoofing(self):
        while self.face_frame.qsize() and self.face_coords.qsize():   
            face_frame = self.face_frame.get()
            image_bbox = self.face_coords.get()
            '''
            # Region uses 2 scaled model FAS
            model_name1='4_0_0_80x80_MiniFASNetV1SE.blob'
            model_name2='2_7_80x80_MiniFASNetV1SE.blob'
            img1=fas(face_frame,image_bbox,model_name1)
            img2=fas(face_frame,image_bbox,model_name2)
            nn_data = run_nn(
                self.anti_spoofing1_in,
                self.anti_spoofing1_nn,
                {"data": to_planar1(img1)},
                )
            nn_data_ = run_nn(
                self.anti_spoofing2_in,
                self.anti_spoofing2_nn,
                {"data": to_planar1(img2)},
                )
            if nn_data is None or nn_data_ is None :
                return False
            result=(softmax(to_nn_result(nn_data))+softmax(to_nn_result(nn_data_)))/2
            '''

            # Region only uses model FAS
            face_frame=face_frame[image_bbox[1] : image_bbox[3], image_bbox[0] : image_bbox[2]]
            img=self.trans(face_frame)
            nn_data__ = run_nn(
                self.anti_spoofing3_in,
                self.anti_spoofing3_nn,
                {"data": to_planar1(img)},
                )
            
            if nn_data__ is None  :
                return False
            result=softmax(to_nn_result(nn_data__))

            label=np.argmax(result)
            if label==1:   
                self.put_text(
                "Real Face ",
                (15,10),
                (0, 0, 255),0.45,
                )  
            elif label!=1:
                self.put_text(
                "Fake Face ",
                (15,10),
                (0, 0, 255),0.45,
                )
            self.face_pro.put(result)          
        return True

    def run_head_pose(self):

        while self.face_frame_reg.qsize():
            face_frame = self.face_frame_reg.get()
            nn_data = run_nn(
                self.head_pose_in,
                self.head_pose_nn,
                {"data": to_planar(face_frame, (60, 60))},
            )
            if nn_data is None:
                return False
            out = np.array(nn_data.getLayerFp16("angle_r_fc"))
            self.face_frame_corr.put(correction(face_frame, -out[0]))
        return True

    def run_arcface(self):

        while self.face_frame_corr.qsize() :
            text=""
            face_frame = self.face_frame_corr.get()         
            face_coords = self.face_coords.get()
            nn_data = run_nn(
                self.insightface_in,
                self.insightface_nn,
                {"data": to_planar(face_frame, (112, 112))},
            )
            if nn_data is None:
                return False
            self.fps_nn.update()
            results = to_nn_result(nn_data)
            t,pro,vec=self.face_recognize(results)
            corr=np.corrcoef(results,vec)
            if pro<self.pro_threshold or corr[0,1]>self.corr_threshold: #0.99 and 0.54
                text=t            
            self.put_text(
                text,
                (face_coords[0], face_coords[1] - 10),
                (0, 0, 255),0.65,
            )   
        return True

    def parse_fun(self):

        time_start=time.time()
        if self.run_face_mn():
            if self.run_anti_spoofing() and self.run_head_pose() :
                if self.run_arcface():
                    time_end=time.time()
                    #print("Time:",time_end-time_start)
                    return True


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera,pro_threshold=args.pro_t,corr_threshold=args.corr_t).run()
