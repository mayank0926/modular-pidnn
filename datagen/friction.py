import gym
import glfw
import numpy as np
from tqdm import tqdm


def datagen(config):
    """Genrates data for frictional domain.

    Uses the pip installed Gym environment in a loop to interface MuJoCo and collect data for the puck released with an initial velocity of vx.

    Args:
        config: Python dictionary representation of the config yaml file.

    Returns:
        Saves a file containing the data generated in a csv format.
        Column headers - [time, initial_velocity, displacement]
    """
    ts = np.arange(0, config['TIME_STEP'] *
                   config['NUM_SNAPSHOTS'], config['TIME_STEP'])
    vxs = np.linspace(config['VX_START'], config['VX_END'], config['VX_NUM'])

    num_cols = config['NUM_INPUTS'] + config['NUM_OUTPUTS']
    data = np.array([]).reshape(0, num_cols)

    # Away from puck, rises up in z
    arm_movement = np.array([-1, -1, 1, 0], dtype=np.float32)

    for vx in tqdm(vxs, desc='Friction data generation progress'):
        obj_init = np.array([vx, 0.0], dtype=np.float32)
        env = gym.make(
            'frictional_pidnn_datagen:mujoco-slide-v0', obj_init=obj_init)
        env.reset()

        # One step allowed to MuJoCo to adjust joint constraints
        obs, reward, done, info = env.step(arm_movement)
        initial_obj_pos = obs["observation"][3:6]
        # Since simulator returns velocity multiplied by dt
        initial_obj_vel = obs["observation"][14:17]/config['TIME_STEP']

        for t in ts:
            if config['render']:
                env.render(mode='human')
            obj_pos = obs["observation"][3:6]
            if (obj_pos[2]-initial_obj_pos[2]) > 0.1:
                # Too much change in y coordinate
                # Did not raise error because ideally this shouldn't impact x direction
                print(f"PUCK FELL OFF THE TABLE! (Velocity {vx})")

            iteration_data = np.array([t, initial_obj_vel, obj_pos])
            data = np.vstack([data, iteration_data])
            obs, reward, done, info = env.step(arm_movement)

        env.close()
        if config['render']:
            glfw.terminate()

    if config['save_collected']:
        np.savetxt(config['datadir'] + config['datafile'], data, delimiter=",")


if __name__ == "__main__":
    print("Not meant to be executed as main!")
    from sys import exit
    exit(1)
