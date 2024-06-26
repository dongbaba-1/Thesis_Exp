import time

import airsim
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

while client.isApiControlEnabled():
    # get the current position of the drone
    position = client.getMultirotorState().kinematics_estimated.position
    # print the position
    print("kinematics_estimated.Position: x={:.2f}, y={:.2f}, z={:.2f}".format(position.x_val, position.y_val, position.z_val))
    time.sleep(1)
    # get the current collision information
    collision_info = client.simGetCollisionInfo()
    # print the collision_info.position
    if collision_info.has_collided:
        print("Collision at position: x={:.2f}, y={:.2f}, z={:.2f}".format(collision_info.position.x_val, collision_info.position.y_val, collision_info.position.z_val))
