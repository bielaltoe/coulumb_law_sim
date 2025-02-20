# Simulador da Lei de Coulomb

Uma simulação interativa de partículas carregadas que demonstra a Lei de Coulomb em um ambiente 3D. O simulador permite visualizar diferentes configurações de cargas elétricas e suas interações.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/coulumb_law_sim.git
cd coulumb_law_sim
```

2. (Recomendado) Crie um ambiente virtual:
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Executando a Simulação

Para iniciar o simulador com a interface gráfica:
```bash
python ui.py
```

## Como Usar

### Controles da Interface

- **Presets**: Use o menu suspenso para escolher diferentes configurações predefinidas
- **Time step (dt)**: Ajuste a velocidade da simulação usando o slider
- **Pause/Resume**: Pause ou continue a simulação
- **Reset**: Reinicie a configuração atual

### Controles da Visualização 3D

- **Rotação**: Clique e arraste com o botão esquerdo do mouse
- **Zoom**: Role a roda do mouse
- **Pan**: Clique e arraste com o botão direito do mouse

### Presets Disponíveis

1. **Orbital**: Sistema complexo com partículas orbitando
2. **Dipole**: Sistema simples com duas cargas opostas
3. **Ring**: Anel de cargas negativas ao redor de uma carga central positiva
4. **Ellipse**: Partículas em movimento elíptico
5. **Spiral**: Configuração em espiral
6. **Random Scatter**: Distribuição aleatória de cargas
7. **Stable Binary**: Sistema binário estável
8. **Stable Circular**: Sistema circular estável

## Notas Técnicas

- As trajetórias das partículas são mostradas com gradiente de opacidade
- O tamanho das partículas é proporcional à sua massa
- Partículas que saem dos limites da simulação são automaticamente desativadas

## Licença

Este projeto está sob a licença MIT.