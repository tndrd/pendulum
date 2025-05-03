from pendulum import *

if __name__ == "__main__":
    fps = 120

    params = Model.Params(3, 2, 1, 1.5, 10, 1e-3)
    state0 = Model.State([0, 0, 2, 6], 0)
    
    model = Model(params, state0)
    view = View(1024, 7)
    contr = Controller(None, None)

    sim = Simulation(fps, model, view, contr)
    sim.start()
