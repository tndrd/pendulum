from pendulum import *

SIMULATION_FPS = 60
WINDOW_SIZE = 768
WINDOW_SCALE = 1.2

# Simulation speedup factor
# Should be integer more or equal 1
SPEEDUP_FACTOR = 3

# Ball masses
BALL1_MASS = 1 # kg
BALL2_MASS = 1 # kg

# Arm lengths
ARM1_LENGTH = 1 # m
ARM2_LENGTH = 1.5 # m

# Gravity acceleration value (aka g)
GRAVITY_G = 10 # m/s^2

# Numeric integration timestep
INTEGRATION_STEP = 1e-3 # s

# Initial system state
INIT_ANGLE_1 = 0
INIT_ANGLE_2 = 0
INIT_SPEED_1 = 4
INIT_SPEED_2 = 4

if __name__ == "__main__":
    params = Model.Params(BALL1_MASS, BALL2_MASS,
                          ARM1_LENGTH, ARM2_LENGTH,
                          GRAVITY_G, INTEGRATION_STEP)
    
    state0 = Model.State([INIT_ANGLE_1, INIT_ANGLE_2,
                          INIT_SPEED_1, INIT_SPEED_2], 0)

    model = Model(params, state0)
    
    view = View(WINDOW_SIZE, 2 * (ARM1_LENGTH + ARM2_LENGTH) * WINDOW_SCALE)

    contr = Controller(None, None)

    sim = Simulation(SIMULATION_FPS, model, view, contr)
    sim.speedup(SPEEDUP_FACTOR)
    sim.start()