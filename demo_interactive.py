from pendulum import *

if __name__ == "__main__":
    fps = 120

    print("\nInteractive mode!\nPress RMB/LMB to interact with pendulum\n")

    params = Model.Params(4, 4, 1, 1, 10, 1e-3)
    state0 = Model.State([0, 0, 2, 0.2], 0)
    
    on_lclick = lambda pos: make_spring_m2(100, view.px2r(pos))
    on_rclick = lambda pos: make_spring_m1(100, view.px2r(pos))

    model = Model(params, state0)
    view = View(1024, 7)
    contr = Controller(on_lclick, on_rclick)

    sim = Simulation(fps, model, view, contr, [make_dissipattor(5, 5)])
    sim.start()
