"""
Módulo de simulação.
Contém a lógica para os cálculos físicos, definição dos presets e a visualização com Vispy.
"""

import os
import numpy as np
from numba import njit
from vispy import app, scene
from vispy.visuals.filters import Alpha, ColorFilter

# Constantes físicas e parâmetros de simulação
k_e = 8.988e9  # Constante de Coulomb
default_dt = 0.005  # Intervalo de tempo padrão

# Define o tipo de dado para as cargas (posição, velocidade, carga, massa e estado ativo)
charge_dtype = np.dtype(
    [
        ("pos", np.float64, 3),  # Posição (x, y, z)
        ("vel", np.float64, 3),  # Velocidade (vx, vy, vz)
        ("q", np.float64),  # Carga elétrica
        ("m", np.float64),  # Massa
        ("active", np.bool_),  # Indica se a carga está ativa
    ]
)

# Presets para diferentes arranjos de partículas
preset_configs = [
    {
        "name": "Orbital",
        "charges": np.array(
            [
                ([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], +8e-6, 5e-2, True),
                ([7.0, 5.0, 5.0], [0.0, 40.0, 0.0], -2e-6, 1e-3, True),
                ([3.0, 5.0, 5.0], [0.0, -40.0, 0.0], -2e-6, 1e-3, True),
                ([7.5, 7.5, 5.0], [-4.242, 4.242, 0.0], -3e-6, 3e-3, True),
                ([2.5, 7.5, 5.0], [-4.242, -4.242, 0.0], -3e-6, 3e-3, True),
                ([2.5, 2.5, 5.0], [4.242, -4.242, 0.0], -3e-6, 3e-3, True),
                ([7.5, 2.5, 5.0], [4.242, 4.242, 0.0], -3e-6, 3e-3, True),
                ([7.0, 5.0, 7.0], [4.242, 0.0, -4.242], +4e-6, 4e-3, True),
                ([3.0, 5.0, 3.0], [-4.242, 0.0, 4.242], +4e-6, 4e-3, True),
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [
                [1.0, 0.8, 0.0, 0.9],  # Central: dourado
                [0.0, 0.5, 1.0, 0.8],  # Orbitas internas: azul claro
                [0.0, 0.5, 1.0, 0.8],  # Orbitas internas: azul claro
                [1.0, 0.0, 0.0, 0.8],  # Orbita média: vermelha
                [0.0, 1.0, 0.0, 0.8],  # Orbita média: verde
                [0.5, 0.0, 0.5, 0.8],  # Orbita média: púrpura
                [1.0, 0.5, 0.0, 0.8],  # Orbita média: laranja
                [0.0, 1.0, 1.0, 0.8],  # Orbita externa: ciano
                [0.0, 1.0, 1.0, 0.8],  # Orbita externa: ciano
            ],
            dtype=np.float32,
        ),
    },
    {
        "name": "Dipole",
        "charges": np.array(
            [
                ([4.0, 5.0, 5.0], [0.0, 0.0, 0.0], +5e-6, 1e-2, True),
                ([6.0, 5.0, 5.0], [0.0, 0.0, 0.0], -5e-6, 1e-2, True),
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [
                [1.0, 0.0, 0.0, 0.9],
                [0.0, 0.0, 1.0, 0.9],
            ],
            dtype=np.float32,
        ),
    },
    {
        "name": "Ring",
        "charges": np.array(
            [
                ([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], +8e-6, 5e-2, True),
                *[
                    (
                        [5.0 + 3 * np.cos(theta), 5.0 + 3 * np.sin(theta), 5.0],
                        [0.0, 0.0, 0.0],
                        -1e-6,
                        1e-3,
                        True,
                    )
                    for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False)
                ],
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [
                [1.0, 0.8, 0.0, 0.9],
                *[[0.0, 1.0, 0.0, 0.8]] * 8,
            ],
            dtype=np.float32,
        ),
    },
    {
        "name": "Ellipse",
        "charges": np.array(
            [([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], +8e-6, 5e-2, True)]
            + [
                (
                    [5.0 + 5 * np.cos(theta), 5.0 + 3 * np.sin(theta), 5.0],
                    [-3 * np.sin(theta), 2 * np.cos(theta), 0.0],
                    -1e-6,
                    1e-3,
                    True,
                )
                for theta in np.linspace(0, 2 * np.pi, 12, endpoint=False)
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [[1.0, 0.8, 0.0, 0.9]] + [[0.2, 0.7, 1.0, 0.8] for _ in range(12)],
            dtype=np.float32,
        ),
    },
    {
        "name": "Spiral",
        "charges": np.array(
            [([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], +8e-6, 5e-2, True)]
            + [
                (
                    [5.0 + theta * np.cos(theta), 5.0 + theta * np.sin(theta), 5.0],
                    [-np.sin(theta), np.cos(theta), 0.0],
                    -1e-6,
                    1e-3,
                    True,
                )
                for theta in np.linspace(0.5, 3 * np.pi, 15)
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [[1.0, 0.8, 0.0, 0.9]] + [[0.9, 0.3, 0.7, 0.8] for _ in range(15)],
            dtype=np.float32,
        ),
    },
    {
        "name": "Random Scatter",
        "charges": np.array(
            [
                # Generate 20 charges scattered randomly in a cube and set random velocities
                *[
                    (
                        [
                            5.0 + 4 * (np.random.rand() - 0.5),
                            5.0 + 4 * (np.random.rand() - 0.5),
                            5.0 + 4 * (np.random.rand() - 0.5),
                        ],
                        [
                            2 * (np.random.rand() - 0.5),
                            2 * (np.random.rand() - 0.5),
                            2 * (np.random.rand() - 0.5),
                        ],
                        (-1e-6 if np.random.rand() > 0.5 else +1e-6),
                        1e-3,
                        True,
                    )
                    for _ in range(20)
                ],
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [*([[np.random.rand(), np.random.rand(), np.random.rand(), 0.8]] * 20)],
            dtype=np.float32,
        ),
    },
    {
        "name": "Stable Binary",
        "charges": np.array(
            [
                ([4.5, 5.0, 5.0], [0.0, 5.0, 0.0], +4e-6, 1e-2, True),
                ([5.5, 5.0, 5.0], [0.0, -5.0, 0.0], -4e-6, 1e-2, True),
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [
                [1.0, 0.3, 0.3, 1.0],
                [0.3, 0.3, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    },
    {
        "name": "Stable Circular",
        "charges": np.array(
            [([5.0, 5.0, 5.0], [0.0, 0.0, 0.0], +8e-6, 5e-2, True)]
            + [
                (
                    [5.0 + 3 * np.cos(theta), 5.0 + 3 * np.sin(theta), 5.0],
                    [-3 * np.sin(theta) * 2, 3 * np.cos(theta) * 2, 0.0],
                    -1e-6,
                    1e-3,
                    True,
                )
                for theta in np.linspace(0, 2 * np.pi, 4, endpoint=False)
            ],
            dtype=charge_dtype,
        ),
        "colors": np.array(
            [[1.0, 0.8, 0.0, 0.9]] + [[0.0, 1.0, 1.0, 0.8]] * 4,
            dtype=np.float32,
        ),
    },
    # Outros presets podem ser adicionados aqui...
]


# Classe principal que gerencia a simulação
class ChargeSimulation:
    def __init__(self):
        # Parâmetros iniciais
        self.dt = default_dt
        self.presets = preset_configs
        self.current_preset = 0
        self.charges = self.presets[self.current_preset]["charges"].copy()
        self.colors = self.presets[self.current_preset]["colors"].copy()
        # Armazena as trajetórias (lista de posições para cada partícula)
        self.trajectories = [[] for _ in range(len(self.charges))]
        self.running = False  # Inicia a simulação pausada

        # Configura os elementos visuais
        self.setup_visuals()
        self._update_visuals_initial()

    def _update_visuals_initial(self):
        # Increase the base size to have larger particles initially
        base_size = 20  # increased base size
        # Also, increase the additional size term for mass effect
        mass_sizes = base_size + 40 * (self.charges["m"] / self.charges["m"].max())
        self.particles.set_data(
            self.charges["pos"],
            edge_color=self.colors,
            face_color=self.colors * 0.7,
            size=mass_sizes,
            edge_width=1.5,
        )

    def setup_visuals(self):
        # Cria a área de visualização (canvas) e a câmera
        self.canvas = scene.SceneCanvas(
            keys="interactive", show=True, bgcolor="#070724", size=(800, 600)
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.fov = 45
        self.view.camera.distance = 15
        self.view.camera.center = (5, 5, 5)

        # Adiciona grade para referência
        self.grid = scene.visuals.GridLines(color=(0.6, 0.6, 0.6, 0.8))
        self.grid.parent = self.view.scene

        # Cria os marcadores para as partículas
        self.particles = scene.visuals.Markers(parent=self.view.scene)
        self.particles.symbol = "disc"
        self.particles.attach(ColorFilter((1, 1, 1, 1)))
        self.particles.attach(Alpha(1))

        # Cria linhas para desenhar trajetórias de cada partícula
        self.traj_lines = [
            scene.visuals.Line(
                color=color, width=4, method="gl", parent=self.view.scene
            )
            for color in self.colors
        ]

        # Timer para atualização contínua da simulação
        self.timer = app.Timer(interval="auto", connect=self.update)

    @staticmethod
    @njit
    def compute_forces(charges):
        # Calcula as forças de Coulomb entre todas as cargas ativas
        n = len(charges)
        forces = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            if not charges[i]["active"]:
                continue
            for j in range(i + 1, n):
                if not charges[j]["active"]:
                    continue
                r = charges[j]["pos"] - charges[i]["pos"]
                dist = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2) + 1e-14
                force_mag = k_e * charges[i]["q"] * charges[j]["q"] / dist**3
                forces[i] += force_mag * r
                forces[j] -= force_mag * r
        return forces

    @staticmethod
    @njit
    def update_physics(charges, forces, dt, boundary):
        # Atualiza a posição e velocidade das partículas de acordo com a física
        for i in range(len(charges)):
            if charges[i]["active"]:
                charges["vel"][i] += forces[i] / charges[i]["m"] * dt
                charges["pos"][i] += charges["vel"][i] * dt
                # Se a partícula sair dos limites definidos, desativa-a
                if np.any(charges["pos"][i] < boundary[0]) or np.any(
                    charges["pos"][i] > boundary[1]
                ):
                    charges["active"][i] = False

    def update(self, event):
        # Atualiza o estado da simulação a cada quadro
        if not self.running:
            return

        forces = self.compute_forces(self.charges)
        self.update_physics(self.charges, forces, self.dt, boundary=(-100000, 100000))

        # Armazena a posição atual de cada partícula na respectiva trajetória
        for i in range(len(self.charges)):
            self.trajectories[i].append(self.charges["pos"][i].copy())

        # Atualiza visualmente os marcadores das partículas, considerando massa e velocidade
        base_size = 20  # increased base size in update as well
        mass_sizes = base_size + 40 * (self.charges["m"] / self.charges["m"].max())
        velocities = np.linalg.norm(self.charges["vel"], axis=1)
        v_norm = velocities / (velocities.max() + 1e-14)
        self.particles.set_data(
            self.charges["pos"],
            edge_color=self.colors,
            face_color=self.colors * 0.7,
            size=mass_sizes,
            
            edge_width=1.5,
        )

        # Atualiza as trajetórias visuais com mudança gradual de opacidade
        for i, line in enumerate(self.traj_lines):
            trail_pos = np.array(self.trajectories[i])
            if len(trail_pos) > 0:
                alphas = np.linspace(0.1, 0.6, len(trail_pos))
                trail_colors = np.ones((len(trail_pos), 4)) * self.colors[i]
                trail_colors[:, 3] = alphas
                line.set_data(trail_pos, color=trail_colors)
                line.visible = True
            else:
                # Ensure that the line remains hidden with a dummy point to avoid empty arrays.
                dummy_pos = np.zeros((1, 3))
                dummy_color = np.zeros((1, 4))
                line.set_data(dummy_pos, color=dummy_color)
                line.visible = False

        self.canvas.update()

    def reset_simulation(self, preset_index=None):
        # Reinicia os atributos da simulação: cargas, trajetórias e cores
        if preset_index is not None:
            self.current_preset = preset_index
        self.charges = self.presets[self.current_preset]["charges"].copy()
        self.colors = self.presets[self.current_preset]["colors"].copy()
        self.trajectories = [[] for _ in range(len(self.charges))]
        # Reinitialize trajectory lines so that their number matches the current charges
        self.traj_lines = [
            scene.visuals.Line(
                color=color, width=4, method="gl", parent=self.view.scene
            )
            for color in self.colors
        ]
        self._update_visuals_initial()


# Se executado diretamente, inicia a simulação sem a camada de UI
if __name__ == "__main__":
    sim = ChargeSimulation()
    sim.running = True
    sim.timer.start()
    app.run()
