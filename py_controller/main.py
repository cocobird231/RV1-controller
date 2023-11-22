import time
import socket
import struct
import random
from enum import Enum

import threading
from threading import Thread

import rclpy
from rclpy.node import Node

from vehicle_interfaces.msg import GPS
from vehicle_interfaces.msg import IMU
from vehicle_interfaces.msg import ControlInfo
from vehicle_interfaces.msg import SteeringWheel

from vehicle_interfaces.srv import ControlReg
from vehicle_interfaces.srv import ControlReq
from vehicle_interfaces.srv import ControlSteeringWheel

from vehicle_interfaces.params import GenericParams
from vehicle_interfaces.vehicle_interfaces import VehicleServiceNode

import math
import numpy as np

import torch
import torch.nn.functional as F

from dataloader import GlobalKNNDataset, RVDatasetType
from models import KNNPredict

class Params(GenericParams):
    def __init__(self, nodeName : str):
        super().__init__(nodeName)
        self.operationMode = 'IDClient'
        self.timeout_ms = 100.0

        self.topic_gps = 'gps_0'
        self.topic_imu = 'imu_0'

        self.trainFilePath = 'datasets/dataset_reverse_global_feat.csv'
        self.modelPath = 'models/localrvnet_best.pth'
        self.knn_k = 2
        self.knn_r = 0.00004

        self.serviceName = 'controller_0'
        self.regServiceName = 'controlserver_0'


        self.declare_parameter('operationMode', self.operationMode)
        self.declare_parameter('timeout_ms', self.timeout_ms)

        self.declare_parameter('topic_gps', self.topic_gps)
        self.declare_parameter('topic_imu', self.topic_imu)

        self.declare_parameter('trainFilePath', self.trainFilePath)
        self.declare_parameter('modelPath', self.modelPath)
        self.declare_parameter('knn_k', self.knn_k)
        self.declare_parameter('knn_r', self.knn_r)

        self.declare_parameter('serviceName', self.serviceName)
        self.declare_parameter('regServiceName', self.regServiceName)
        self._getParam()

    def _getParam(self):
        self.operationMode = self.get_parameter('operationMode').get_parameter_value().string_value
        self.timeout_ms = self.get_parameter('timeout_ms').get_parameter_value().double_value

        self.topic_gps = self.get_parameter('topic_gps').get_parameter_value().string_value
        self.topic_imu = self.get_parameter('topic_imu').get_parameter_value().string_value

        self.trainFilePath = self.get_parameter('trainFilePath').get_parameter_value().string_value
        self.modelPath = self.get_parameter('modelPath').get_parameter_value().string_value
        self.knn_k = self.get_parameter('knn_k').get_parameter_value().integer_value
        self.knn_r = self.get_parameter('knn_r').get_parameter_value().double_value

        self.serviceName = self.get_parameter('serviceName').get_parameter_value().string_value
        self.regServiceName = self.get_parameter('regServiceName').get_parameter_value().string_value


class SteeringWheelController(VehicleServiceNode):# Devinfo, Timesync
    def __init__(self, params : Params):
        super().__init__(params)

        # Store params
        self.__params = params
        self.__exitF = False

        # Pytorch
        self.__trainLoader = GlobalKNNDataset(csvPath=params.trainFilePath, loaderType=RVDatasetType.GLOBAL_SIMP)
        self.__net = KNNPredict(self.__trainLoader.getDataset(), k=params.knn_k, r=params.knn_r)
        self.get_logger().info('[SteeringWheelController] KNN modules loaded (k:%d, r:%f).' %(params.knn_k, params.knn_r))

        # Subscription
        self.__gpsSub = self.create_subscription(GPS, params.topic_gps, self.__gpsCallback, 10)
        self.__imuSub = self.create_subscription(IMU, params.topic_imu, self.__imuCallback, 10)
        self.__currentGPS = GPS()
        self.__currentIMU = IMU()

        # Register to control server
        self.__regClientNode = Node(params.nodeName + '_controlreg_client')
        self.__regClient = self.__regClientNode.create_client(ControlReg, params.regServiceName + "_Reg")
        while (not self.__regToControlServer()) : time.sleep(0.5)# Try registration until success.
        self.get_logger().info('[SteeringWheelController] Register to control server.')

        # SteeringWheelController service
        self.__server = self.create_service(ControlSteeringWheel, params.serviceName, self.__serverCallback)
        self.get_logger().info('[SteeringWheelController] Controller server created.')

        # Control signal
        self.__currentControlSignal = None
        self.__controlSignalFrameId = 0

        # Run KNN prediction
        self.__predTh = threading.Thread(target=self.__calSteeringWheel)
        self.__predTh.start()
    
    def __del__(self):
        self.__predTh.join()
    
    def __calSteeringWheel(self):
        while (not self.__exitF):
            lat = self.__currentGPS.latitude
            lon = self.__currentGPS.longitude
            orient = self.__currentIMU.orientation

            pred = self.__net.predict(torch.tensor([lat, lon], dtype=torch.float))
            # Predicted data
            steer = 0 if (torch.isnan(pred[0])) else pred[0].item()
            thr = 0 if (torch.isnan(pred[1])) else pred[1].item()
            ret = SteeringWheel()
            ret.gear = ret.GEAR_REVERSE
            ret.steering = int(steer)
            ret.pedal_throttle = int(thr)

            self.__currentControlSignal = ret
            time.sleep(0.05)

    def __regToControlServer(self):
        myInfo = ControlInfo()
        myInfo.service_name = self.__params.serviceName
        myInfo.msg_type = myInfo.MSG_TYPE_STEERING_WHEEL
        myInfo.timeout_ms = float(self.__params.timeout_ms)

        request = ControlReg.Request()
        request.request = myInfo

        future = self.__regClient.call_async(request)
        rclpy.spin_until_future_complete(self.__regClientNode, future, timeout_sec=0.1)
        if (not future.done()):
            self.get_logger().error('[SteeringWheelController.__regToControlServer] Failed to call service.')
        else:
            response = future.result()
            if (response.response):
                return True
        return False

    def __serverCallback(self, request, response):# ControlSteeringWheel.srv
        response.response = False
        if (request.request):
            if (self.__currentControlSignal):
                response.value = self.__currentControlSignal
                response.frame_id = self.__controlSignalFrameId
                self.__controlSignalFrameId += 1
                response.response = True
        return response
    
    def __gpsCallback(self, msg):
        self.__currentGPS = msg
    
    def __imuCallback(self, msg):
        self.__currentIMU = msg

def main(args=None):
    rclpy.init(args=args)
    params = Params('controller_params_node')
    server = SteeringWheelController(params)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(server)
    executorTH = threading.Thread(target=executor.spin, daemon=True)
    executorTH.start()
    executorTH.join()
    rclpy.shutdown()
