import rospy
import time
import tf
import numpy as np
import copy

#ros message related
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Int8
from std_srvs.srv import Empty

class StageWorld():
    def __init__(self, robot_index, goal_size=0.5):
        self.robot_index = robot_index
        node_name = 'env_' + str(robot_index)
        rospy.init_node(node_name, anonymous=None)

        #laser
        self.laser_mum = 512
        self.laser_index =  0
        self.scan = None

        #initial world
        self.goal = [0., 0.]
        self.self_speed = [0.0, 0.0]
            #self.step_r_cut = 0

        #generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)#20*20m
        self.goal_size = goal_size

            # self.robot_value = 10.
            # self.goal_value = 0.

        self.init_pose = None

            #self.stop_counter = 0

        # ----|service|----
        self.reset = rospy.ServiceProxy('reset_positions', Empty)

        # ----|Publisher|----
        topic_cmd_vel = 'robot_' + str(robot_index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(topic_cmd_vel, Twist, queue_size=10)
        
        topic_cmd_pose = 'robot_' + str(robot_index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(topic_cmd_pose, Pose, queue_size=10)
        
        # ----|Subscriber|----    http://wiki.ros.org/stage_ros 
        #get robot real position
        topic_odom_state = 'robot_'+str(robot_index)+'/base_pose_ground_truth'
        self.sub_odom_state = rospy.Subscriber(topic_odom_state, Odometry, self.base_pose_ground_truth_callback)

        topic_odom = 'robot_' + str(robot_index) + '/odom'
        self.sub_odom = rospy.Subscriber(topic_odom, Odometry, self.odometry_callback)

        topic_laser = 'robot_' + str(robot_index) + '/base_scan'
        self.sub_laser =  rospy.Subscriber(topic_laser, LaserScan, self.laser_sacn_callback)

        topic_clock = 'clock'
        self.sub_clock = rospy.Subscriber(topic_clock, Clock, self.clock_callback)

        topic_int8 = 'robot_' + str(robot_index) + '/int8'
        self.sub_check_crash = rospy.Subscriber(topic_int8, Int8, self.crach_callback)


        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        # wait until the first date callback
  
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:       
                # print('sacn:', self.scan)
                # print('speed:', self.speed)
                # print('state:', self.state)
                # print('speed_GT:', self.speed_GT)
                # print('state_GT:', self.state_GT)

                pass

        #rospy.sleep(1)

    def base_pose_ground_truth_callback(self, odometry):
        quat = odometry.pose.pose.orientation
        #Quaternious
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.state_GT = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, euler[2]]
        vel_x = odometry.twist.twist.linear.x
        vel_y = odometry.twist.twist.linear.y
        velocity = np.sqrt(vel_x**2 + vel_y**2)
        self.speed_GT = [velocity, odometry.twist.twist.angular.z]

    
    def odometry_callback(self, odometry):
        quat = odometry.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def laser_sacn_callback(self, laserscan):
        self.scan_param = [laserscan.angle_min, laserscan.angle_max, laserscan.angle_increment\
                           ,laserscan.time_increment, laserscan.range_min, laserscan.range_max]
        self.scan = np.array(laserscan.ranges)
        self.laser_index += 1

    def clock_callback(self, clock):
        self.time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crach_callback(self, flag):
        self.is_crashed = flag.data


    # get variable
    def get_state_GT(self):
        return self.state_GT

    def get_speed_GT(self):
        return self.speed_GT

    def get_state(self):
        return self.stage

    def get_speed(self):
        return self.speed

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(sacn)] = 6.0
        scan[np.isinf(scan)] = 6.0
        scan_leaght = len(scan)
        sparse_laser_num = self.laser_mum
        step = float(scan_leaght) / sparse_laser_num
        
        sparse_laser_left = []
        sparse_laser_right = []
        index_left = 0
        index_right = scan_leaght-1

        for i in xrange(int(sparse_beam_num/2)):
            sparse_laser_left.append(scan[int(index_left)])
            sparse_laser_right.append(scan[int(index_right)])
            index_left += step
            index_right -= step
        sparse_laser = np.concatenate((sparse_laser_left, sparse_laser_right[::-1]), axis = 0)
        print(sparse_laser)
        print('----------------')
        print(sparse_laser / 6.0 - 0.5)
        return sparse_laser / 6.0 - 0.5

    def get_time(self):
        return self.time

    def get_crash_state(self):
        return self.is_crashed

    def get_local_goal():
        [x, y, theta] = self.get_state_GT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = (goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def generate_goal_point(self):
        [goal_x, goal_y] = self.generate_random_goal()
        self.goal_point = [goal_x, goal_y]
        [x, y] = self.get_local_goal()
        self.pre_distance = np.sqrt(x**2 + y**2)
        self.distance = copy.deepcopy(self.pre_distance)

    def generate_random_goal(self):
        self.init_pose = self.get_state_GT()
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis_origin = np.sqrt(x**2 + y**2) 
        dis_goal =  np.sqrt((x - self.init_pose[0])**2 + (y - self.init_pose[1])**2)
        while (dis_origin > 9 or dis_goal > 10 or dis_goal < 8) and not rospy.is_shutdown():
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis_origin = np.sqrt(x**2 + y**2) 
            dis_goal =  np.sqrt((x - self.init_pose[0])**2 + (y - self.init_pose[1])**2)
        return [x, y]

    def get_reward_and_terminate(self):
        terminate = 1
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_state_GT()
        [v, w] = self.get_speed_GT()
        self.pre_distance =  np.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x)**2 + (self.goal_point[1] - y)**2)
        
        #build reward ---------> init
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result =0
        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = 1
            reward_g = 15
            result = 'reach goal'
        
        if is_crash == 1:
            terminate =1
            reward_c = -15
            result = 'crashed'

        if np.abs(w) > 1.05:
            reward_w = -0.1*np.abs(w)

        if t >150:
            terminate = 1
            result = 'time out'

        reward = reward_c + reward_g +reward_w

        return reward, terminate, result
        
    def reset_world(self):
        self.reset()
        self.self_speed = [0.0, 0.0]
        self.goal = [0., 0.]
        self.step_r_cut = 0
        self.start_time =time.time()
        rospy.sleep(0.5)

    def generate_random_pose(self):
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis = np.sqrt(x**2 + y**2)
        while (dis > 9) and not rospy.is_shutdown():
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis = np.sqrt(x**2 + y**2)
        theta = np.random.uniform(0, 2*np.pi)
        #print(x,y,dis)
        return [x, y, theta]
        
        
    def control_pose(self,pose):
        pose_cmd = Pose()
        assert len(pose) == 3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.x = qtn[1]
        pose_cmd.orientation.x = qtn[2]
        pose_cmd.orientation.x = qtn[3]
        self.cmd_pose.publish(pose_cmd)


    def reset_pose(self):
        random_pose = self.generate_random_pose()
        print(1)
        rospy.sleep(0.01)
        self.control_pose(random_pose)
        print(2)

        [robot_x, robot_y, theta] = self.get_state_GT()
        print(3)

        while np.abs(random_pose[0] - robot_x) > 0.2 or np.abs(random_pose[1]-robot_y) > 0.2:
            [robot_x, robot_y, theta] = self.get_state_GT()
            self.control_pose(random_pose)
            print(np.abs(random_pose[0] - robot_x),np.abs(random_pose[1]-robot_y))
        print(4)

        rospy.sleep(0.01)

    def control_vel(self, action):
        move_cmd =Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0
        move_cmd.linear.z = 0
        move_cmd.angular.x = 0
        move_cmd.angular.y = 0
        move_cmd.angular.z = action[1]

        self.cmd_vel.publish(move_cmd)
