import math
import os
import random
import subprocess
import time
from os import path
import math
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y): # Prevent starting position from overlapping with obstacles
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0
        self.goal_received = False

        self.upper = 5.0 # Upper bound for goal position generation
        self.lower = -5.0 # Lower bound for goal position generation
        self.initial_pose = {
            "x": 0.0,
            "y": -1.5,
            "z": 0.01,
            "yaw": -0.3,
        }
        self.velodyne_data = np.ones(self.environment_dim) * 10 # Create an array of environment dimension with all elements as 10
        self.last_odom = None

        self.set_self_state = ModelState() # Initialize the robot
        self.set_self_state.model_name = "p3dx"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]] # Divide 180 degrees into 20 intervals
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        # port = "11311"
        # subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True) # Initialize node
        # if launchfile.startswith("/"):
        #     fullpath = launchfile
        # else:
        #     fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        # if not path.exists(fullpath):
        #     raise IOError("File " + fullpath + " does not exist")

        # subprocess.Popen(["roslaunch", "-p", port, fullpath])
        # print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/p3dx/cmd_vel", Twist, queue_size=1) # Publish velocity
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        ) # Publish robot initial position information
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty) # When calling this service, unpause physics simulation in Gazebo, continue running
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty) # When calling this service, pause physics simulation in Gazebo
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty) # When calling this service, reset Gazebo simulation world to initial state
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3) # Publish visualization marker array for goal point
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1) # Publish visualization marker array for linear velocity
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1) # Publish visualization marker array for angular velocity
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        ) # Subscribe to Velodyne LiDAR sensor point cloud data and pass to velodyne_callback method
        self.odom = rospy.Subscriber(
            "/p3dx/odom", Odometry, self.odom_callback, queue_size=1
        ) # Subscribe to robot's position, orientation, linear velocity and angular velocity
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=1
        ) # Subscribe to manually set goal point

    def goal_callback(self, goal_msg):
        """Handle manual goal updates from RViz or CLI publishing."""
        self.goal_x = goal_msg.pose.position.x
        self.goal_y = goal_msg.pose.position.y
        self.goal_received = True

        # allow immediate visualization update when goal changes
        self.publish_markers([0.0, 0.0])

    def wait_for_manual_goal(self):
        """Block until a manual goal is provided on /move_base_simple/goal."""
        if self.goal_received:
            return

        rospy.loginfo("Waiting for goal on /move_base_simple/goal...")
        goal_msg = rospy.wait_for_message("/move_base_simple/goal", PoseStamped)
        self.goal_callback(goal_msg)

    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v): 
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z"))) # Read point cloud data from message v, extract x, y, z coordinates of each point, field_names indicates only extracting x, y, z coordinates of each point
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2: # Filter out points lower than 0.2 meters below the sensor
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2)) # Square root of (x squared + y squared)
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1]) # Calculate angle between point (x, y) and x-axis
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2) # Calculate distance from point to origin

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action, last_distance, last_action, last_x, last_y):
        done_target = False
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics") # Pause program execution until service "/gazebo/unpause_physics" becomes available
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds Stop for a short time, let the robot drive for a period of time at the current speed, TIME_DELTA = 0.1
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data) # Determine if collision occurred through minimum data from LiDAR
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False) # Convert quaternion to Euler angles
        angle = round(euler[2], 4) # Extract rotation angle around z-axis (i.e., robot's heading) from Euler angles, round to 4 decimal places and store in angle

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        self_distance = np.linalg.norm(
            [self.odom_x - last_x, self.odom_y - last_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal Calculate relative angle between robot and goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        if mag1 == 0 or mag2 == 0:
            beta = 0.0
        else:
            value = max(min(dot / (mag1 * mag2), 1.0), -1.0)
            beta = math.acos(value) # Inverse cosine to calculate angle between robot's current heading and heading toward goal
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle # Calculate relative angle between robot's current heading and heading toward goal
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
            done_target = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, last_action, min_laser, last_distance, distance)
        return state, reward, done, target, distance, done_target, collision, self_distance

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy() # Call service to reset Gazebo simulation environment

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        self.goal_received = False

        quaternion = Quaternion.from_euler(
            0.0, 0.0, self.initial_pose["yaw"]
        ) # Use fixed initial heading
        object_state = self.set_self_state # Initialize robot node

        object_state.pose.position.x = self.initial_pose["x"] # Set fixed initial position
        object_state.pose.position.y = self.initial_pose["y"]
        object_state.pose.position.z = self.initial_pose["z"]
        object_state.pose.orientation.x = quaternion.x # Set fixed initial heading
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state) # Publish robot's state

        self.odom_x = object_state.pose.position.x # Set robot's initial position
        self.odom_y = object_state.pose.position.y

        self.wait_for_manual_goal()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics") # Resume program execution
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA) # Execute for a period of time

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        if self.last_odom is None:
            try:
                self.last_odom = rospy.wait_for_message(
                    "/p3dx/odom", Odometry, timeout=1.0
                )
            except rospy.ROSException:
                pass

        v_state = []
        v_state[:] = self.velodyne_data[:] # Copy LiDAR data
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        ) # Calculate straight-line distance between robot's current position and goal position

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        if mag1 == 0 or mag2 == 0:
            beta = 0.0
        else:
            value = max(min(dot / (mag1 * mag2), 1.0), -1.0)
            beta = math.acos(value)

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - self.initial_pose["yaw"]

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0] # Contains robot's state information
        state = np.append(laser_state, robot_state) # Merge LiDAR state and robot state
        return state, distance, self.initial_pose["x"], self.initial_pose["y"] # Return initial state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10: 
            self.upper += 0.004 # Possibly to gradually expand upper bound of goal position
        if self.lower > -10:
            self.lower -= 0.004 # Possibly to gradually expand lower bound of goal position

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self): # Randomly place boxes
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y) # Check if box position is valid
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y]) # Calculate distance between box and robot
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y]) # Calculate distance between box and goal position
                if distance_to_robot < 1.5 or distance_to_goal < 1.5: # If box is too close to robot or goal position, position is invalid
                    box_ok = False
            box_state = ModelState() # Create a ModelState object to set box state
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action): # Publish visualization data in rviz
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod # Indicates observe_collision is a static method
    def observe_collision(laser_data): # Detect collision from LiDAR data
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser 
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, last_action, min_laser, last_distance, distance):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0]*(last_distance - distance) / 2 - abs(action[1] - last_action[1]) / 2 - r3(min_laser) / 2 
