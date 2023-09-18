import time
import math
import numpy as np
from isaacgym import gymapi, gymutil
from setup.xMonsterKinematics import *

xmk = HexapodKinematics()

offset = 0.1

h = 0.2249
T = 10
L_step = 0.15

l1 = 0
l2 = 3.25
l3 = 3.25

a = l1 + l2*np.cos(0.52) + l3*np.sin(1.57)
b = l2*np.sin(0.52) - l3*np.cos(1.57)
c = -0.03

phi = 4*math.pi/T                    
A = L_step/(np.sin(phi*T/2) - phi*T/2)
theta = 0;

gym = gymapi.acquire_gym()

args = gymutil.parse_arguments(description="Yuna CPG")

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 100
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

asset_root = "resources/robots/yuna"
yuna_asset_file = "urdf/yuna.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
asset = gym.load_asset(sim,asset_root,yuna_asset_file)

# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)

# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# gym.set_actor_dof_states(env,actor_handle,dof_states,gymapi.STATE_POS)

def move145(direction, current):
    Q = np.array([[L_step,  L_step,  L_step,   L_step, L_step, L_step],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]])
    rotatedPosition = current + direction*Q
    rotatedPosition = [rotatedPosition[:, 0], current[:, 1], current[:, 2], rotatedPosition[:, 3], rotatedPosition[:, 4], current[:, 5]]
    rotatedPosition = np.transpose(rotatedPosition)
    rotatedPosition = np.array(rotatedPosition)
    return rotatedPosition
    
def move236(direction, current):
    Q = np.array([[L_step,  L_step,  L_step,   L_step, L_step, L_step],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]])
    rotatedPosition = current + direction*Q
    rotatedPosition = [current[:, 0],rotatedPosition[:, 1], rotatedPosition[:, 2], current[:, 3], current[:, 4], rotatedPosition[:, 5]]
    rotatedPosition = np.transpose(rotatedPosition)   
    rotatedPosition = np.array(rotatedPosition)
    return rotatedPosition

## Walk Functions ##    
def swing145(direction, current):
    currentPos = current
    t = 0
    if (direction == 'forward'):
        theta = 0
        mul = 1
    elif (direction == 'backward'):
        theta = 0
        mul = - 1
    elif (direction == 'left'):
        theta = pi/2
        mul = - 1
    elif (direction == 'right'):
        theta = -pi/2
        mul = 1
        
    while (t <= 10):
        dis = A*(phi*t - np.sin(phi*t))
        vel = A*(phi - phi*cos(phi*t))
        acc = A*phi*phi*sin(phi*t)
        
        x = -dis * np.cos(theta)
        y = -dis*np.sin(theta)
        z = -(a * dis**2 + b*dis + c)

        xMovement = mul* x * np.ones(6)
        yMovement = y * np.ones(6)
        zMovement = z * np.ones(6)
        
        swingArray = np.vstack((xMovement/2, yMovement/2, zMovement))
        moveArray = np.vstack((-xMovement/2, -yMovement/2, np.zeros(6)))
        swingPos = eePos + swingArray
        movePos = eePos + moveArray

        currentPos = [swingPos[:, 0], movePos[:, 1], movePos[:, 2], swingPos[:, 3], swingPos[:, 4], movePos[:, 5]]
        currentPos = np.transpose(currentPos)
        currentPos = np.array(currentPos)
        command_ang = legCorrectAng(currentPos)
        frame = updateRender(command_ang)
        t = t + 1

def swing236(direction, current):
    currentPos = current
    t = 0
    if (direction == 'forward'):
        theta = 0
        mul = 1
    elif (direction == 'backward'):
        theta = 0
        mul = - 1
    elif (direction == 'left'):
        theta = pi/2
        mul = - 1
    elif (direction == 'right'):
        theta = -pi/2
        mul = 1
        
    while (t <= 10):
        dis = A*(phi*t - np.sin(phi*t))
        vel = A*(phi - phi*cos(phi*t))
        acc = A*phi*phi*sin(phi*t)
        
        x = -dis * np.cos(theta)
        y = -dis*np.sin(theta)
        z = -(a * dis**2 + b*dis + c)

        xMovement = mul* x * np.ones(6)
        yMovement = y * np.ones(6)
        zMovement = z * np.ones(6)
        
        swingArray = np.vstack((xMovement/2, yMovement/2, zMovement))
        moveArray = np.vstack((-xMovement/2, -yMovement/2, np.zeros(6)))
        swingPos = eePos + swingArray
        movePos = eePos + moveArray
        
        currentPos = [movePos[:, 0],swingPos[:, 1], swingPos[:, 2], movePos[:, 3], movePos[:, 4], swingPos[:, 5]]
        currentPos = np.transpose(currentPos)
        currentPos = np.array(currentPos)
        command_ang = legCorrectAng(currentPos)
        frame = updateRender(command_ang)
        t = t + 1

## Helper Functions ##    
def legsDown(current, h):
    updatedPosition = current
    updatedPosition[2] = [-h, -h, -h,  -h, -h,  -h]
    command_ang = legCorrectAng(updatedPosition)
    frame = updateRender(command_ang, 0, 5)
    return updatedPosition
    
def lift145up(current, h):
    legUp = current
    legUp[2] = [(-h + offset), -h, -h,  -h + offset, -h + offset,  -h]
    return legUp
    
def lift236up(current, h):
    legUp = current
    legUp[2] =[-h, (-h + offset), -h + offset, -h,  -h, -h + offset]
    return legUp

def updateRender(targetPos):
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    targetPos = targetPos.astype(np.float32)
    gym.set_actor_dof_position_targets(env,actor_handle,targetPos)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

def legCorrectAng(rotatedPosition):
    rotatedAng = xmk.getLegIK(rotatedPosition)
    #correction for legs
    commanded_position = np.zeros(18)
    commanded_position[0:3] = rotatedAng[0:3]     # 1 to 1
    commanded_position[3:6] = rotatedAng[6:9]     # 2 to 3
    commanded_position[6:9] = rotatedAng[12:15]   # 3 to 5
    commanded_position[9:12] = rotatedAng[3:6]    # 4 to 2
    commanded_position[12:15] = rotatedAng[9:12]   # 5 to 4
    commanded_position[15:18] = rotatedAng[15:18]   # 6 to 6
    return commanded_position
    
def walk(direction, stancePos, cycles): 
    i = 0   
    while (i < cycles) :
        currentPos = stancePos
        swing145(direction, currentPos)
        swing236(direction, currentPos)
        i = i + 1
    command_ang = legCorrectAng(stancePos)
    frame = updateRender(command_ang)

eePos = np.array([[0.51589,  0.51589,  0.0575,   0.0575, - 0.45839, - 0.45839],
                   [0.23145, - 0.23145,   0.5125, - 0.5125,   0.33105, - 0.33105],
                   [-h, -h, -h, -h, -h, -h]])

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.5)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
envs = []
actor_handles = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
# env = gym.create_env(sim,env_lower,env_upper,0)
# actor_handle = gym.create_actor(env,asset,pose,"actor")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "move")
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_states.dtype = gymapi.DofState.dtype
gym.set_actor_dof_states(env,actor_handle,dof_states,gymapi.STATE_POS)
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(1000)
props["damping"].fill(200)
gym.set_actor_dof_properties(env, actor_handle, props)
leg_pos = np.array([0,0,-1.5708,0,0,1.5708,0,0,-1.5708,0,0,1.5708,0,0,-1.5708,0,0,1.5708],dtype=np.float32)
motion_data = []
file_name = "Yuna_walk_test.txt"
file = open(file_name,'w')

step = 1
while not gym.query_viewer_has_closed(viewer):
    
    # gym.simulate(sim)
    # gym.fetch_results(sim, True)

    # # update the viewer
    # gym.step_graphics(sim);
    # gym.draw_viewer(viewer, sim, True)
    # gym.set_actor_dof_position_targets(env,actor_handle,leg_pos)

    # # Wait for dt to elapse in real time.
    # # This synchronizes the physics simulation with the rendering rate.
    # gym.sync_frame_time(sim)
    # if step % 50 == 0:
    walk('forward',eePos,2)
    
    step += 1
    state = gym.get_actor_dof_states(env,actor_handle,3)
    file.write(','.join(map(str,np.array(state))) + '\n')
    # motion_data.append(state)

# np.save("Yuna_train_data",motion_data)




