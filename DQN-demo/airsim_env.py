import airsim
import numpy as np
import cv2
from PIL import Image


class AirSimEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        return self.get_state()

    def get_depth_image(client):
        # 向AirSim请求深度图像
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)])
        response = responses[0]

        # 将深度图像数据转换为一维数组
        img1d = np.array(response.image_data_float, dtype=np.float32)

        # 将深度值归一化到0-255范围内
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)

        # 将一维数组转换为二维图像
        img2d = np.reshape(img1d, (response.height, response.width))

        # 将二维图像转换为灰度图像并取反
        image = np.invert(np.array(Image.fromarray(img2d).convert('L')))

        # 调整图像大小为84x84像素
        image = cv2.resize(image, (84, 84))

        # 将图像转换为浮点数组
        image = np.array(image, dtype=np.float32)

        return image

    def step(self, action):
        if action == 0:
            self.client.moveByVelocityAsync(1, 0, 0, 1).join()
        elif action == 1:
            self.client.moveByVelocityAsync(-1, 0, 0, 1).join()
        elif action == 2:
            self.client.moveByVelocityAsync(0, 1, 0, 1).join()
        elif action == 3:
            self.client.moveByVelocityAsync(0, -1, 0, 1).join()
        elif action == 4:
            self.client.moveByVelocityAsync(0, 0, -1, 1).join()
        elif action == 5:
            self.client.moveByVelocityAsync(0, 0, 1, 1).join()

        next_state = self.get_state()
        collision_info = self.client.simGetCollisionInfo()
        reward = -1 if collision_info.has_collided else 1
        done = collision_info.has_collided
        return next_state, reward, done

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
