### script for lane keeping with semantic segmentation camera
### not usable for now !!!!

def step(self,action):
    return obs, reward, done, extra_info


import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import tensorflow as tf
from collections import deque
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import TensorBoard
#import keras.backend.tensorflow_backend as backend
from threading import Thread
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 20_000
MIN_REPLAY_MEMORY_SIZE = 5_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 100
MODEL_NAME = "ModelTest01"
#MEMORY_FRACTION = 0.6 ## SOLVING RTX MEMORY ERROR
MIN_REWARD = -200

EPISODES = 100
DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.99995 ## 0.9975 or so later
MIN_EPSILON = 0.02

AGGREGATE_STATS_EVERY = 10

# Own Tensorboard class

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
    
    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()
                
class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]



    def reset(self):
        self.collision_hist = []
        self.actor_list = []
####spawn car changed

        world = self.client.get_world()

        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=1.0)

        self.vehicle_blueprint = self.client.get_world().get_blueprint_library().filter('model3')[0]

        self.filtered_waypoints = []
        for waypoint in self.waypoints:
            if(waypoint.road_id == 52):
                self.filtered_waypoints.append(waypoint)

        self.spawn_point = self.filtered_waypoints[14].transform
        self.spawn_point.location.z += 2
        self.transform = self.spawn_point
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        
        self.rgb_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        #self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        #self.radar = self.blueprint_library.find("sensor.other.radar")
        #self.lidar = self.blueprint_library.find("sensor.lidar.ray_cast")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img_rgb(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(0.2)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self,event):
        self.collision_hist.append(event)

    def process_img_rgb(self,image):
        #i = np.array(image.raw_data)
        #print(i.shape)
        #i2 = i.reshape((self.im_height, self.im_width, 4))
        #i3 = i2[:, :, :3]
        #if self.SHOW_CAM:
         #   cv2.imshow("", i3/255)
          #  cv2.waitKey(1)
        #self.front_camera = i3
        image.save_to_disk("/home/edvard/Carla/CARLA_0.9.11/PythonAPI/examples/segcam/testcam.jpg", carla.ColorConverter.CityScapesPalette)
        image = mpimg.imread(
            '/home/edvard/Carla/CARLA_0.9.11/PythonAPI/examples/segcam/testcam.jpg')
        color1 = np.array([70,180,160])
        color2 =np.array([80,255,255])
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_mask = cv2.inRange(image_hsv, color1, color2)
        #image_mask = cv2.cvtColor(image_mask,cv2.COLOR_HSV2BGR)
        self.result = cv2.bitwise_and(image, image, mask = image_mask)
        cv2.imshow("",self.result)
        cv2.waitKey(50)
        self.front_camera = self.result
        #cv2.imwrite("/home/edvard/Carla/CARLA_0.9.11/PythonAPI/examples/segcam/testcam.png", i3/255)
        return self.result


    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=3):
        # If there are no lines to draw, exit.
        if lines is None:
            return    # Make a copy of the original image.
        # Create a blank image that matches the original in size.
        img = np.copy(img)
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )    # Loop over all lines and draw them on the blank image.
        result = self.result
        for line in lines:
            for x1, y1, x2, y2 in line: # Merge the image with the lines onto the original.
                cv2.line(line_img, (int(x1), int(y1)),(int(x2), int(y2)), color, thickness)  # Return the modified image.
        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

        lines = cv2.HoughLinesP(
            result,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )


        self.left_line_x = []
        self.left_line_y = []
        self.right_line_x = []
        self.right_line_y = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
                if math.fabs(slope) < 0.8:  # <-- Only consider extreme slope
                    continue
                if slope <= 0:  # <-- If the slope is negative, left group.
                    self.left_line_x.extend([x1, x2])
                    self.left_line_y.extend([y1, y2])
                else:  # <-- Otherwise, right group.
                    self.right_line_x.extend([x1, x2])
                    # <-- Just below the horizon
                    self.right_line_y.extend([y1, y2])

        min_y = result.shape[0] * (3 / 5)
        max_y = result.shape[0]

        poly_left = np.poly1d(np.polyfit(
            self.left_line_y,
            self.left_line_x,
            deg=1
        ))
        self.left_x_start = int(poly_left(max_y))
        self.left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            self.right_line_y,
            self.right_line_x,
            deg=1
        ))

        self.right_x_start = int(poly_right(max_y))
        self.right_x_end = int(poly_right(min_y))
        #return 

    def get_line_image(self):
        self.line_image = self.draw_lines(
            self.result,
            [[
                [self.left_x_start, self.max_y, self.left_x_end, self.min_y],
                [self.right_x_start, self.max_y, self.right_x_end, self.min_y],
            ]],
            thickness=5,
        )
        return self.line_image

    def get_steering_angle(self, ):
        pass
        #return

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer =-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action ==2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200

        elif kmh < 15:
            done = False
            reward = -1
       # elif kmh > 60:
        #    done = False
         #   reward = -1
        else:
            done = False
            reward = 1
        
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            reward = 100
            done = True

        return self.front_camera, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        #self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0

        self.training_initialized = False

    def create_model(self):
       
       
        #base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
#
 #       x = base_model.output
  #      x = GlobalAveragePooling2D()(x)
   #     predictions = Dense(3, activation="linear")(x)
    #    model = Model(inputs = base_model.input, outputs = predictions)
     #   model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
      #  return model
       
        model = Sequential()

        model.add(Conv2D(64, (5, 5), input_shape=(
            IM_HEIGHT, IM_WIDTH, 3), padding='same'))

        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5),
                strides=(3, 3), padding='same'))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5),
                strides=(3, 3), padding='same'))
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(3, 3),
                strides=(2, 2), padding='same'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(3, 3),
                strides=(2, 2), padding='same'))
        model.add(Flatten())
        #return model.input, model.output
        ## model_head_hidden_dense
        inputs = [model.input]
        x = model.output
        x = Dense(3, activation='relu')(x)
        # And finally output (regression) layer
        predictions = Dense(3, activation='linear')(x)
        # Create a model
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(
            lr=0.001), metrics=["accuracy"])
        return model




    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)
    
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        #with self.graph.as_default():
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states=np.array([transition[0] for transition in minibatch])/255
        #with self.graph.as_default():
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step
        
        #with self.graph.as_default():
        self.model.fit(np.array(x)/255, np.array(y, batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False))

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH,3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        #with self.graph.as_default():
        self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True
        
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-200]

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    env = CarEnv()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)
    
    agent.get_qs(np.ones((env.im_height, env.im_width,3)))

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episodes"):
        env.collision_hist =  []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else: 
                action = np.random.randint(0, 3)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            episode_reward += reward
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            step += 1

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()

            
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate=True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
