define laser ranger
(
  sensor(
    pose [ 0 0 0.1 0 ]
    fov 180
    range [ 0.0 6.0 ]
    samples 512
  )
  color "random"
  block( 
    points 4
    point[0] [0 0]
    point[1] [0 1]
    point[2] [1 1]
    point[3] [1 0]
    z [0 0.21]
  )

)


define agent position
(
    # actual size
    #size [0.44 0.38 0.22] # sizes from MobileRobots' web site
    size [0.45 0.45 0.22]
    # the pioneer's center of rotation is offset from its center of area
    origin [0 0 0 0]

    # draw a nose on the robot so we can see which way it points
    gui_nose 1

    color "random"
    drive "diff"		 	# Differential steering model.
    obstacle_return 1           	# Can hit things.
    ranger_return 0.5            	# reflects sonar beams
    blob_return 1               	# Seen by blobfinders  
    fiducial_return 1           	# Seen as "1" fiducial finders
    laser(
        pose [ 0 0 0 0 ] 
    )
    
    # [ xmin xmax ymin ymax zmin zmax amin amax ]				
    # velocity_bounds [-0.5 0.5 0 0 0 0 -90.0 90.0 ]					
    # acceleration_bounds [-0.5 0.5 0 0 0 0 -90 90.0 ]
  )