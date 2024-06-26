import airsim
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# use API to takeoff and land the drone
client.enableApiControl(True)
# unlock
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
print("home point = ", client.getHomeGeoPoint())
print("vehicle pose = ", client.simGetVehiclePose())
print("object pose = ", client.simGetObjectPose(''))
print("collision.position = ", client.getMultirotorState().collision.position)
print("kinematics_estimated.position = ", client.getMultirotorState().kinematics_estimated.position)
print("rotor states = ", client.getRotorStates())
# print(client.getDistanceSensorData())
client.takeoffAsync().join()

client.moveToZAsync(-8, 1).join()
# client.moveToPositionAsync(10, 0, -10, 1).join()
client.moveByVelocityAsync(10,0,0,5).join()
# client.moveByVelocityZAsync(1,1,-12,10).join()
# client.moveByVelocityBodyFrameAsync(1,1,-1,10).join()
client.moveToPositionAsync()
# client.reset()
client.landAsync().join()

# lock
client.armDisarm(False)
# # release control
# client.enableApiControl(False)