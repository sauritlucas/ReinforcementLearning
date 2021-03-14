import gym
import numpy as np
import random


class TatetiEnv(gym.Env):
    """
    Ambiente personalizado que sigue la interfaz de gym.
    Es un entorno simple en el cuál el agente debe aprender a jugar ta-te-ti
    """
    # Dado que estamos en colab, no podemos implementar la salida por interfaz
    # gráfica ('human' render mode)
    metadata = {'render.modes': ['console']}
    # constantes
    CLEAN_SQUARE = 0
    AGENT_CHIP = 1
    PLAYER_CHIP = 2

    """
    Tablero de tateti
      +-----------+
      | 7 | 8 | 9 |
      +-----------+
      | 4 | 5 | 6 |
      +-----------+
      | 1 | 2 | 3 |
      +-----------+
    """
    WIN_COMB = [
         {6, 7, 8},
         {3, 4, 5},
         {0, 1, 2},
         {0, 3, 6},
         {1, 4, 7},
         {2, 5, 8},
         {2, 4, 6},
         {0, 4, 8}
    ]
    INITIAL_STATE = np.array([0] * 9, dtype=np.int32)

    def __init__(self):
        super(TatetiEnv, self).__init__()

        # Tamaño de la grilla de 2D
        self.grid_size = 9
        # Inicializamos en agente a la derecha de la grilla
        self.agent_pos = self.INITIAL_STATE.copy()
        # Definimos el espacio de acción y observaciones
        # Los mismos deben ser objetos gym.spaces
        # En este ejemplo usamos dos acciones discretas: izquierda y derecha
        n_actions = 9
        self.action_space = gym.spaces.Discrete(n_actions)
        # La observación será el estado del tablero
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(9,),
            dtype=np.int32
        )

    def reset(self) -> np.ndarray:
        """
        Importante: la observación devuelta debe ser un array de numpy
        :return: (np.array)
        """
        # Se inicializa el agente a la derecha de la grilla
        self.agent_pos = self.INITIAL_STATE.copy()
        # convertimos con astype a float32 (numpy) para hacer más general el agente
        # (en caso de que querramos usar acciones continuas)
        return self.agent_pos

    def step(self, action: int):
        if 0 <= action < 9:
            if self.agent_pos[action] == self.CLEAN_SQUARE:
                self.agent_pos[action] = self.AGENT_CHIP
                casilla_ocupada = False
            else:
                casilla_ocupada = True #dice que la casilla estaba ocupada para que no cambie el estado
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # gano el agente o el oponente? si gano alguno, termina la partida
        # 0 -> no hay ganador, 1 -> gana el agente, 2 -> gana el oponente, 3 -> finaliza sin ganadores
        if any(self.WIN_COMB[j].issubset(
                {i for i, value in enumerate(self.agent_pos) if value == self.AGENT_CHIP}
            )
            for j in range(len(self.WIN_COMB))
        ):
            winner = 1
        elif len([i for i, value in enumerate(self.agent_pos) if value == self.CLEAN_SQUARE]) == 0 :# si no hay celdas libres
            winner = 3
        else:
            winner = 0

        # el oponente elige una posicion al asar de las que quedan libres
        if winner == 0 and not casilla_ocupada:
            self.agent_pos[
                random.choice(
                    [i for i, value in enumerate(self.agent_pos) if value == self.CLEAN_SQUARE]
                )
            ] = self.PLAYER_CHIP

        if any(self.WIN_COMB[j].issubset(
                {i for i, value in enumerate(self.agent_pos)if value == self.PLAYER_CHIP}
            )
            for j in range(len(self.WIN_COMB))
        ):
            winner = 2

        # Asignamos recompensa sólo cuando el agente llega a su objetivo
        # (recompensa = 0 en todos los demás estados)
        rewards = {
            #<winner>: reward
            0: -1,
            1: 200,
            2: -20,
            3: -15
        }
        reward = rewards[winner] if not casilla_ocupada else 0

        # gym también nos permite devolver información adicional, ej. en atari:
        # las vidas restantes del agente (no usaremos esto por ahora)
        info = {}

        return self.agent_pos, reward, bool(winner), info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # en nuestra interfaz de consola, representamos el agente con 'x', el oponente con 'o' y vacio con ' '
        symbols = {
            self.CLEAN_SQUARE: ' ',
            self.AGENT_CHIP: 'X',
            self.PLAYER_CHIP: 'O'
        }
        to_print = [symbols[s] for s in self.agent_pos]

        print('+-----------+')
        for i in range(3):
                print(f"| {to_print[3*i]} | {to_print[3*i+1]} | {to_print[3*i+2]} |")
        print('+-----------+\n')


    def close(self):
        pass