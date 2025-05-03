import numpy as np
import pygame
import time
from solvers import RungeKuttaCollection, CALC_TYPE


class Model:
    class State:
        # Phase vector of a system
        # [p1, p2, w1, w2]
        # p1, p2 - angles
        # w1, w2 - d(p1)/dt, d(p2)/dt
        def __init__(self, data, t):
            # p1, p2, w1, w2
            self.data = np.array(data, CALC_TYPE)
            self.t = float(t)

        def to_tuple(self): return *self.data, self.t

        def print(self):
            print(f"State:\n"
                  f"  p1: {self.p1}\n"
                  f"  p2: {self.p2}\n"
                  f"  w1: {self.w1}\n"
                  f"  w2: {self.w2}\n")

    class Params:
        M_UPPER_BOUND = 10
        EPSILON = 1e-3
        
        M_CONSTRAINTS = [EPSILON, M_UPPER_BOUND]

        L_CONSTRAINTS = [EPSILON, 5]
        G_CONSTRAINTS = [0, 100]

        DT_CONSTRAINTS = [1e-4, 100]

        def __init__(self, m1, m2, l1, l2, g, dt):
            self.m1 = m1
            self.m2 = m2
            self.l1 = l1
            self.l2 = l2
            self.g = g
            self.dt = dt
            
            self.validate()

        def to_tuple(self):
            return self.m1, self.m2, self.l1, self.l2, self.g, self.dt

        @staticmethod
        def _validate1(x, bounds):
            mn, mx = bounds
            return x >= mn and x <= mx
        
        def validate(self):
            values = self.m1, self.m2, self.l1, self.l2, self.g, self.dt
            bounds = self.M_CONSTRAINTS, self.M_CONSTRAINTS, self.L_CONSTRAINTS, self.L_CONSTRAINTS, self.G_CONSTRAINTS
            names = ("m1", "m2", "l1", "l2", "g", "dt")

            for val, bound, name in zip(values, bounds, names):
                if not self._validate1(val, bound):
                    raise RuntimeError(f"Parameter \"{name}\" has incorrect value\n"
                                       f"Expected {bound[0]} <= {name} <= {bound[1]}, got {val}")

    def __init__(self, params: Params, state0: State):
        self.state = state0
        self.params = params
        self.solver = RungeKuttaCollection.create_e4(params.dt)
        self.forces = []
        self.solver.init_problem(lambda q, t: self._ODE_RFunc(params, q, t, self.forces), state0.t, state0.data)

    def update(self, interactors):
        for interactor in interactors:
            interactor.mhandle(self)
        
        self.state.t, self.state.data = self.solver.step()
        self.forces = []

    def get_xy(self):
        p1, p2 = self.state.data[0:2]
        r1 = self.params.l1 * np.array([np.sin(p1), -np.cos(p1)], dtype=CALC_TYPE)
        dr2 = self.params.l2 * np.array([np.sin(p2), -np.cos(p2)], dtype=CALC_TYPE)
        r2 = r1 + dr2

        return r1, r2

    @staticmethod
    def _Q_spring_m1(q, params, k, center):
        xc, yc = center
        p1 = q[0]
        Q0 = k * params.l1 * (xc*np.cos(p1) + yc*np.sin(p1))
        return np.array([Q0, 0], dtype=CALC_TYPE)

    @staticmethod
    def _Q_spring_m2(q, params, k, center):
        xc, yc = center
        p1, p2 = q[0], q[1]

        Q0 = k * params.l1 * (xc*np.cos(p1) + yc*np.sin(p1) + params.l2 * np.sin(p1 - p2))
        Q1 = k * params.l2 * (xc*np.cos(p2) + yc*np.sin(p2) + params.l1 * np.sin(p2 - p1))

        return np.array([Q0, Q1], dtype=CALC_TYPE)

    def apply_spring_m1(self, k, center):
        self.forces.append(lambda q: self._Q_spring_m1(q, self.params, k, center))

    def apply_spring_m2(self, k, center):
        self.forces.append(lambda q: self._Q_spring_m2(q, self.params, k, center))

    def apply_friction(self, b1, b2):
        self.forces.append(lambda q: self._Q_dissipate(q, b1, b2))

    @staticmethod
    def _Q_dissipate(q, b1, b2):
        _, __, w1, w2 = q
        Q = np.empty(2, dtype=CALC_TYPE)

        Q[0] = -(b1 + b2) * w1
        Q[1] = -b2 * w2

        return Q        

    @staticmethod
    def _ODE_RFunc(params, q, t, forces):
        p1, p2, w1, w2 = q

        m1, m2, l1, l2, g, _ = params.to_tuple()

        dp1 = w1
        dp2 = w2

        Q = (0, 0) 
        if len(forces) != 0:
            Q = np.sum([force(q) for force in forces], axis=0)

        f1 = -m2*l2*(w2**2)*np.sin(p1 - p2) - g*(m1 + m2)*np.sin(p1) + Q[0] / l1
        f2 = m2*l1*(w1**2)*np.sin(p1 - p2) - g*m2*np.sin(p2) + Q[1] / l2
        a = (f1 - f2*np.cos(p1 - p2))/(m1 + m2*(np.sin(p1 - p2)**2))

        dw1 = a / l1
        dw2 = (f2/m2 - a*np.cos(p1 - p2)) / l2

        return np.array([dp1, dp2, dw1, dw2], dtype=CALC_TYPE)

class View:
    BALL_R_MIN = .1
    BALL_R_MAX = .25

    LINE_WIDTH_M = 5e-2

    def __init__(self, wsize_px, wsize_m):
        self.wsz_px = wsize_px
        self.wsz_m = wsize_m

        pygame.init()
        self.screen = pygame.display.set_mode((wsize_px, wsize_px))

    def _r2px(self, r):
        mirror = np.array([1, -1])
        c = np.array([self.wsz_m, self.wsz_m], CALC_TYPE) / 2
        return np.round((c + r * mirror) * (self.wsz_px / self.wsz_m), 0).astype(np.int32)
    
    def px2r(self, px):
        rpx = np.array(px, dtype=CALC_TYPE)
        c = np.array([self.wsz_px, self.wsz_px], CALC_TYPE) / 2
        rpx -= c
        rpx[1] *= -1

        return rpx * (self.wsz_m / self.wsz_px)
            
    def _line(self, r1, r2, color, width):
        pygame.draw.line(self.screen, color, r1, r2, width)

    def _circ(self, x, r, color):
        pygame.draw.circle(self.screen, color, x, r)

    def draw(self, model: Model, interactors):
        params = model.params

        # Drawing circles
        self.screen.fill((255, 255, 255))

        r1, r2 = model.get_xy()

        rad1 = self._ballmass2radius(params.m1, params.M_CONSTRAINTS)
        rad2 = self._ballmass2radius(params.m2, params.M_CONSTRAINTS)

        r1px = self._r2px(r1)
        r2px = self._r2px(r2)

        #print(r1px, r2px)

        lw = self._line_width_px(self.LINE_WIDTH_M)

        self._line(self._r2px(0), r1px, (0, 0, 0), lw)
        self._line(r1px, r2px, (0, 0, 0), lw)
        
        self._circ(r1px, rad1, (255, 0, 0))
        self._circ(r2px, rad2, (0, 0, 255))
        self._circ(self._r2px(0), lw, (0, 0, 0))

        for interactor in interactors:
            interactor.vhandle(self, model)

        for event in pygame.event.get(): pass
        pygame.display.update()

    def draw_spring_m1(self, model: Model, k, center):
        r1, _ = model.get_xy()
        self._line(self._r2px(r1), self._r2px(center), (0, 0, 0), 10)

    def draw_spring_m2(self, model: Model, k, center):
        _, r2 = model.get_xy()
        self._line(self._r2px(r2), self._r2px(center), (0, 0, 0), 10)

    def _ballmass2radius(self, mass, mbounds):
        sqrt_mmin = np.sqrt(mbounds[0])
        sqrt_mmax = np.sqrt(mbounds[1])
        sqrt_m = np.sqrt(mass)

        a = (self.BALL_R_MAX - self.BALL_R_MIN) / (sqrt_mmax - sqrt_mmin)
        b = self.BALL_R_MAX - a * sqrt_mmax

        return np.round((a * sqrt_m + b) * self.wsz_px / self.wsz_m, 0).astype(np.uint32)
    
    def _line_width_px(self, w):
        return np.round(w * self.wsz_px / self.wsz_m).astype(int)

class Interactor:
    def __init__(self, mhandle, vhandle):
        self.mhandle = mhandle
        self.vhandle = vhandle

def make_spring_m1(k, center):
    mhandler = lambda model: model.apply_spring_m1(k, center)
    vhandler = lambda view, model: view.draw_spring_m1(model, k, center)
    return Interactor(mhandler, vhandler)

def make_spring_m2(k, center):
    mhandler = lambda model: model.apply_spring_m2(k, center)
    vhandler = lambda view, model: view.draw_spring_m2(model, k, center)
    return Interactor(mhandler, vhandler)

def make_dissipattor(b1, b2):
    mhandler = lambda model: model.apply_friction(b1, b2)
    vhandler = lambda view, model: None
    return Interactor(mhandler, vhandler)

class Controller:
    def __init__(self, mouse_lclick_handler, mouse_rclick_handler,):
        self.mouse_lclick_handler = mouse_lclick_handler
        self.mouse_rclick_handler = mouse_rclick_handler

    def process_events(self):
        interactors = []
        pressed = pygame.mouse.get_pressed()
        pos = pygame.mouse.get_pos()
        if pressed[0] and self.mouse_lclick_handler is not None:
            interactors.append(self.mouse_lclick_handler(pos))
        if pressed[2] and self.mouse_lclick_handler is not None:
            interactors.append(self.mouse_rclick_handler(pos))
        return interactors
        

class Simulation:
    def __init__(self, fps, model: Model, view: View, controller: Controller, interactors=[]):
        self.model = model
        self.view = view
        self.controller = controller
        self.fps = fps
        self.rtime = self.model.state.t
        self.interactors = interactors

    def _compose_interactors(self, int2):
        newint = []
        for intr in self.interactors:
            newint.append(intr)
        for intr in int2:
            newint.append(intr)
        return newint

    def start(self):
        steps_per_frame = np.round(1/(self.fps * self.model.params.dt)).astype(int)

        while True:
            t_start = time.perf_counter()             

            intrs = self._compose_interactors(self.controller.process_events())
            
            for i in range(steps_per_frame):
                self.model.update(intrs)
            
            self.view.draw(self.model, intrs)

            t_end = time.perf_counter()

            sleep_time = 1/self.fps - (t_end - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

            t_end_2 = time.perf_counter()

            print(f"fps: {int(1 / (t_end_2 - t_start))}", end = "\r")