# -*- coding: UTF-8 -*-
from tscil.agents.base import SequentialFineTune
from tscil.agents.l2p import L2P

agents = {
    'SFT': SequentialFineTune,
    'L2P': L2P
          }
