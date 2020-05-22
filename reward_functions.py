import numpy as np

def rf1(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    # print("Dist: %s"%dist)
    reward = -dist - dist2
    return reward

def rf2(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    # print("Dist: %s"%dist)
    reward = 1/dist  + 1/dist2
    return reward

def rf3(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    # print("Dist: %s"%dist)
    reward = 1/(dist  + dist2)
    return reward

def rf4(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    # print("Dist: %s"%dist)
    reward = 1/(dist  + dist2)**2
    return reward

def rf5(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    # print("Dist: %s"%dist)
    reward = -dist**2 - dist2
    return reward

def rf6(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])

    v1 = (goal - state[3:])/dist
    v2 = (state[3:] - state[:3])/dist2

    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    ang = np.absolute(np.arctan2(sinang, cosang))
    # print("Dist: %s"%dist)
    reward = -dist**2 - dist2 -ang**3
    return reward

def rf7(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    reward = -0.1
    if(dist<0.075):
        reward = 100
    return reward

def rf8(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    v1 = (goal - state[3:])/dist
    v2 = (state[3:] - state[:3])/dist2

    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    ang = np.absolute(np.arctan2(sinang, cosang))
    # print("Dist: %s"%dist)
    reward = -dist - dist2 - ang
    return reward

def rf9(state,goal):
    dist = (np.linalg.norm(goal - state[3:]))
    dist2 = np.linalg.norm(state[3:] - state[:3])
    v1 = (goal - state[3:])/dist
    v2 = (state[3:] - state[:3])/dist2

    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    ang = np.absolute(np.arctan2(sinang, cosang))
    # print("Dist: %s"%dist)
    reward = -dist - dist2 - ang/10
    return reward

def rrf1(state,goal):
    dist = (np.linalg.norm(goal - state))
    reward = -dist
    return reward

def rrf2(state,goal):
    dist = (np.linalg.norm(goal - state))
    reward = -dist**2
    return reward

def rrf3(state,goal):
    dist = (np.linalg.norm(goal - state))
    reward = -0.1
    if(dist<0.075):
        reward = 100
    return reward

def rrf4(state,goal):
    dist = (np.linalg.norm(goal - state))
    reward = 1/dist
    return reward

def rrf5(state,goal):
    dist = (np.linalg.norm(goal - state))
    reward = 1/dist**2
    return reward

def rrf6(state,goal):
    dist = (np.linalg.norm(goal - state))
    reward = -0.1
    if(dist < 0.2):
        reward = 1/dist**2
    return reward

pusher_rfs = [rf1,rf2,rf3,rf4,rf5,rf6,rf7,rf8,rf9]
pusher_rfs_str = [ "reward = -dist - dist2",
                   "reward = 1/dist  + 1/dist2",
                   "reward = 1/(dist  + dist2)",
                   "reward = 1/(dist  + dist2)**2",
                   "reward = -dist**2 - dist2",
                   "reward = -dist**2 - dist2 - ang**3",
                   "reward = (dist<0.075)?100 : -0.1",
                   "reward = -dist - dist2 - ang",
                   "reward = -dist**3 - dist2**2 - ang/10"]

reacher_rfs = [rrf1,rrf2,rrf3,rrf4,rrf5,rrf6]

reacher_rfs_str = [
    "reward = -dist",
    "reward = -dist**2",
    "(dist<0.075)?100:-0.1",
    "reward = 1/dist",
    "reward = 1/dist**2",
    "(dist<0.2)?(1/dist**2):-0.1"
]

reward_functions = {'pusher':{},'reacher':{}}
reward_functions['pusher']['rfs']=pusher_rfs
reward_functions['pusher']['str']=pusher_rfs_str
reward_functions['reacher']['rfs']=reacher_rfs
reward_functions['reacher']['str']=reacher_rfs_str