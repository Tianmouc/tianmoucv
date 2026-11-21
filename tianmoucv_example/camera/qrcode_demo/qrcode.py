#coding=utf-8

import numpy as np
import threading
import time

import cv2
import torch

#算法
from tianmoucv.proc.reconstruct import sd_gradients, poisson_blending, sd_gray_preview
#start camera and read
from tianmoucv.camera.sdk_utils import TMstreamingDataReader
from tianmoucv.camera.sdk_utils import global_aop_queue,global_cop_queue,result_queue

import cv2
from pyzbar import pyzbar


import argparse


# Create the parser
parser = argparse.ArgumentParser(description='choose camera id')

# Add arguments to the parser
parser.add_argument('--id', type=int, default=0,help='input file')

# Parse the arguments
args = parser.parse_args()

# deal with [AOP] data and put the result back to result queue
def AOP_thread():
    while True:
        time.sleep(0.01)
        if global_aop_queue.qsize() > 0:
            #tsd
            rawdata = global_aop_queue.get()
            td,sdl,sdr = rawdata
            rawtsdiff = np.stack([td,sdl,sdr],axis=-1)
            rawtsdiff[abs(rawtsdiff)<3] = 0
            #result_queue.put(('aop',showimg/32.0))
            #gray recon
            gx, gy = sd_gradients(np.stack((sdl, sdr)))
            gray_int8 = sd_gray_preview(poisson_blending(gx, gy))
            gray_3 = np.repeat(gray_int8[..., None], 3, axis=-1) / 255.0
            qr_codes = pyzbar.decode(gray_int8)

            imshow_core = np.concatenate([rawtsdiff/128.0,gray_3],axis=1)
            imshow = np.zeros([320,640,3])
            height, width = imshow_core.shape[:2]
            biash, biasw = (320-height)//2, (640-width)//2
            imshow[biash:biash+height,biasw:biasw+width,:] = imshow_core
            imshow = (imshow*255).astype(np.uint8)
            

            for qr_code in qr_codes:
                (x, y, w, h) = qr_code.rect
                x += biasw + rawtsdiff.shape[1]
                y += biash
                cv2.rectangle(imshow, (x, y), (x + w, y + h), (0, 0, 255), 2)
                qr_code_data = qr_code.data.decode("utf-8")
                qr_code_type = qr_code.type
                cv2.putText(imshow, qr_code_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            imshow = imshow.astype(np.float32)/255.0
            result_queue.put(('AOP',imshow))
    

# deal with [COP] data and put the result back to result queue
def COP_thread():
    while True:
        time.sleep(0.01)
        if global_cop_queue.qsize() > 0:
            #RGB show
            rawdata,rawdata_rod = global_cop_queue.get()
            #fuse HDR
            if not rawdata_rod is None:
                rgb = rawdata.astype(np.float32)
                td,sdl,sdr = rawdata_rod
                gx, gy = sd_gradients(np.stack((sdl, sdr)).astype(np.float32) / 128)
                blend_hdr = poisson_blending(gx, gy, srcimg=rgb / 255.0, mask_rgb=True)
                blend_hdr = blend_hdr.numpy() * 255.0
                imshow = np.concatenate([rgb,blend_hdr],axis=0) 
                imshow[imshow>255]=255
                imshow[imshow<0]=0

                imshow_uint8 = imshow.astype(np.uint8)
                qr_codes = pyzbar.decode(np.mean(imshow_uint8,axis=-1))
                for qr_code in qr_codes:
                    (x, y, w, h) = qr_code.rect
                    cv2.rectangle(imshow_uint8, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    qr_code_data = qr_code.data.decode("utf-8")
                    qr_code_type = qr_code.type
                    cv2.putText(imshow_uint8, qr_code_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                imshow = imshow_uint8.astype(np.float32)

                result_queue.put(('COP',imshow[...,[2,1,0]]/255.0))
        
######################################################
#   main loop
######################################################
cfg_path = None #"/home/lyh/projects/lyncam_proj/lyncam_driver/cfg_csv/para_8bit/Golden_HCG_seq.csv"
lib_path = None #"/home/lyh/projects/tianmoucv/cpplib/libtianmouc.so"
tianmoucReader = TMstreamingDataReader(cfg_path=cfg_path,lib_path=lib_path,device_id=args.id)

#update exposure time, in us
tianmoucReader.setExposure(aoptime=1200,coptime=15000,aopG=1,copG=1)

MAX_ROD_BUFFER = 100#protect memory

with tianmoucReader as reader:
    count = 0
    threadaop = threading.Thread(target=AOP_thread)
    threadcop = threading.Thread(target=COP_thread)
    threadaop.daemon = True  # Set the thread as a daemon thread
    threadcop.daemon = True  # Set the thread as a daemon thread
    threadcop.start()
    threadaop.start()

    rod_buffer = dict([])

    while True:
        isrod, timestamp, rawdata = reader()

        #push to rod thread
        if isrod:
            rod_buffer[timestamp] = rawdata
            count += 1
            if count % 12 == 0:
                global_aop_queue.put(rawdata)

        #push to cone thread
        else:
            min_gap = 1e99
            best_ts = -1
            nearestRod = None
            for ts in rod_buffer:
                if timestamp>ts:
                    gap = timestamp-ts
                else:
                    gap = ts-timestamp
                if gap < min_gap:
                    best_ts = ts
                    min_gap = gap
            
            nearestRod = rod_buffer[best_ts]
            if timestamp-best_ts > 1400 or len(rod_buffer)>MAX_ROD_BUFFER:
                rod_buffer = dict([])

            #print('frame',timestamp,best_ts,'  gap:',timestamp-best_ts)
            global_cop_queue.put((rawdata,nearestRod))
            
        if result_queue.qsize() > 0:
            result_name,data = result_queue.get()
            cv2.imshow(result_name, data)
            cv2.waitKey(1)
        

