SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
MAX_EPISODE = 10


class CarEnvDistanceReward:
    metadata = {'render.modes': ['human']}
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self, reward_function=None, verbose=0, EXPERIENCE_SECONDE = None):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.actor_list = []
        self.action_space = Discrete(3)
        self.observation_space = Box(0, 255, [IM_HEIGHT,IM_WIDTH,3])
        self.info = {'episode':0}
        self.reward_function = reward_function
        self.verbose = verbose
        self.iter = 0
        self.experience_seconde = EXPERIENCE_SECONDE

    def reset(self):
        
        self.collision_hist = []
        self.actor_list = []
        self.info = {'episode':0}
        self.iter = 0
        
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        not_spawn = True
        while not_spawn:
            try :
                self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
                not_spawn = False
            except Exception:
                not_spawn = True
            
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        #time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        self.initial_Location = self.vehicle.get_location()
        
        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        #kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -50
            if self.verbose == 1:
                print("Collision detected !")
        else :
            done = False
            distance = self.vehicle.get_location().distance(self.initial_Location)
            if self.reward_function == None : 
                reward = distance
            else:
                reward = self.reward_function(distance)
            if self.verbose == 1:
                print("Obtain "+str(reward)+" reward, distance = "+str(distance))
        
        if self.experience_seconde != None :
            if time.time()-self.episode_start > self.experience_seconde:
                done = True
        else :
            if self.iter > MAX_EPISODE-1 :
                done = True
            self.iter += 1
        self.info['episode'] += 1
        
        return self.front_camera , reward, done, self.info