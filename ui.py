"""
Módulo de interface do usuário (UI) para a simulação.
Separa a camada de visão (PyQt5) da parte dos cálculos/física.
"""

import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QComboBox,
)
from PyQt5 import QtCore

# Importa a classe de simulação definida no módulo simulation.py
from simulation import ChargeSimulation, preset_configs, default_dt


# Define a janela principal que integra a UI e a simulação
class MainWindow(QMainWindow):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.setWindowTitle("Simulação de Cargas - UI Embutida")
        # Incorpora o canvas da simulação na janela principal
        self.setCentralWidget(self.sim.canvas.native)
        self.createOverlayControls()

    def createOverlayControls(self):
        # Cria um widget overlay para controles
        self.control_widget = QWidget(self.centralWidget())
        self.control_widget.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.control_widget.setAttribute(QtCore.Qt.WA_AlwaysStackOnTop)
        self.control_widget.setStyleSheet(
            """
            background-color: rgba(30, 30, 40, 200);
            border-radius: 10px;
            padding: 8px;
            """
        )

        # Layout para os controles
        control_layout = QHBoxLayout(self.control_widget)
        control_layout.setContentsMargins(8, 8, 8, 8)

        # Combobox para seleção de presets
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([p["name"] for p in self.sim.presets])
        self.preset_combo.setStyleSheet("color: white;")  # Set font color to white
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        control_layout.addWidget(self.preset_combo)

        # Controle de tempo de simulação (dt) com slider
        self.dt_label = QLabel(f"Time step (dt): {self.sim.dt:.4f}")
        self.dt_label.setStyleSheet("color: white;")
        control_layout.addWidget(self.dt_label)

        self.dt_slider = QSlider(QtCore.Qt.Horizontal)
        self.dt_slider.setRange(1, 20)
        self.dt_slider.setValue(int(self.sim.dt * 2000))
        self.dt_slider.valueChanged.connect(self.update_dt)
        control_layout.addWidget(self.dt_slider)

        # Botão para pausar/resumir a simulação
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet("color: white;")  # White font
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_btn)

        # Botão para resetar a simulação
        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet("color: white;")  # White font
        reset_btn.clicked.connect(
            lambda: self.sim.reset_simulation(self.sim.current_preset)
        )
        control_layout.addWidget(reset_btn)

        # Ajusta a posição do widget de controles na janela
        QtCore.QTimer.singleShot(0, self.adjustControlPosition)

    def adjustControlPosition(self):
        # Posiciona o widget de controles na parte inferior da janela
        self.control_widget.setGeometry(10, self.height() - 70, self.width() - 20, 60)

    def resizeEvent(self, event):
        # Redimensiona e reposiciona os controles ao mudar o tamanho da janela
        self.adjustControlPosition()
        super().resizeEvent(event)

    def load_preset(self, index):
        # Carrega o preset selecionado
        self.sim.reset_simulation(index)

    def update_dt(self, value):
        # Atualiza o intervalo de tempo (dt)
        self.sim.dt = value / 2000.0
        self.dt_label.setText(f"Time step (dt): {self.sim.dt:.4f}")

    def toggle_pause(self):
        # Alterna entre pausar e retomar a simulação
        self.sim.running = not self.sim.running
        self.pause_btn.setText("Resume" if not self.sim.running else "Pause")


if __name__ == "__main__":
    # Cria a simulação e inicia seu timer
    sim = ChargeSimulation()
    sim.timer.start()

    # Cria a aplicação PyQt5 e a janela principal com a UI embutida
    qt_app = QApplication(sys.argv)
    window = MainWindow(sim)
    window.resize(1000, 800)
    window.show()
    sys.exit(qt_app.exec_())
