### TO MANUALLY FIND WAYPOINTS ON ANY CARLA MAP

import glob
import os
import sys
import random
import time
import numpy as np
import cv2


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla



client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

world = client.get_world()
def draw_waypoints(waypoints, road_id=None, life_time=50.0):

  for waypoint in waypoints:

    if(waypoint.road_id == road_id):
      world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                   persistent_lines=True)


waypoints = client.get_world().get_map().generate_waypoints(distance=1.0)
draw_waypoints(waypoints, road_id=52, life_time=20)

vehicle_blueprint = client.get_world().get_blueprint_library().filter('model3')[0]

filtered_waypoints = []
for waypoint in waypoints:
    if(waypoint.road_id == 52):
      filtered_waypoints.append(waypoint)

spawn_point = filtered_waypoints[14].transform
spawn_point.location.z += 2
vehicle = client.get_world().spawn_actor(vehicle_blueprint,spawn_point)
print(type(spawn_point))

'''
   def draw_waypoints(waypoints, road_id=None, life_time=50.0):
       self.world = self.client.get_world()
       for waypoint in self.waypoints:

            if (waypoint.road_id == road_id):
                self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                             color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                             persistent_lines=True)

        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=1.0)
        self.draw_waypoints(waypoints, road_id=30, life_time=20)
        filtered_waypoints = []

        for waypoint in waypoints:
            if(waypoint.road_id == 30):
                filtered_waypoints.append(waypoint)

        spawn_point = Transform(Location(x=1.445265, y=45.555687, z=2.000000), Rotation(
            pitch=360.000000, yaw=269.637451, roll=0.000000))
        #spawn_point.location.z += 2
        #print(spawn_point)
'''
