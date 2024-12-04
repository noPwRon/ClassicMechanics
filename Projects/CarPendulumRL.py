import matplotlib.pyplot as plt

# math and simulation management
import numpy as np
import jax
import jax.numpy as jnp
from brax.training.agents.ppo import train as ppo

# 3D rendering
import mujoco
import mediapy

import collimator
from collimator import library
from collimator.optimization import RLEnv

from IPython.display import clear_output


class CartPole(collimator):
    def __init__(
        self,
        x0=jnp.zeros(4),
        m_c=float,
        m_p=float,
        L=float,
        a=float,
        name="CartPole",
    ):
        super().__init__(name=name)
        self.declare_dynamic_parameter("m_c", m_c)
        self.declare_dynamic_parameter("m_p", m_p)
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("g", a)

        self.declare_input_port(name="fx")
        self.declare_continuous_state(default_value=x0, ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, theta, dot_x, dot_theta = state.continuous_state
        (fx,) = inputs

        m_c = parameters["m_c"]
        m_p = parameters["m_p"]
        L = parameters["L"]
        g = parameters["g"]

        # below is the equation of motion for the cart and pendulum

        mf = 1.0 / (m_c + m_p * jnp.sin(theta) ** 2)
        ddot_x = mf * (
            fx + m_p * jnp.sin(theta) * (L * dot_theta**2 + g * jnp.cos(theta))
        )

        ddot_theta = (
            (1.0 / L) * mf * (-fx * jnp.cos(theta))
            - m_p * L * dot_theta**2 * jnp.cos(theta) * jnp.sin(theta)
            - (m_c + m_p) * g * jnp.sin(theta)
        )

        return jnp.array([dot_x, dot_theta, ddot_x[0], ddot_theta[0]])

    # using the simulate interface to simulate the environment and system

    system = CartPole(x0=np.array([0.0, 3 * np.pi / 4, 0.0, 0.0]))
