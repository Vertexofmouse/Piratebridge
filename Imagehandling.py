# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 02:03:00 2022

@author: Nagatorinne
"""

import os
import pygame
from PIL import Image
        
    
    
class table_images():
    def __init__(self, M, n, window_size=(1700,854)):
        self.width_ratio = window_size[0]/1700
        self.height_ratio = window_size[1]/854
        self.M = M
        self.n = n
        self.K = {'spades':0, 'hearts':1, 'clubs':2, 'diamonds':3}
        path_deck = os.path.join(os.getcwd(), 'deck.PNG')
        path_table = os.path.join(os.getcwd(), 'table%s.PNG'%str(self.M))
        path_token = os.path.join(os.getcwd(), 'Turntoken.PNG')
        path_backside = os.path.join(os.getcwd(), 'backside.PNG')
        path_trickbackside = os.path.join(os.getcwd(), 'trickbackside.PNG')
        self.deck = Image.open(path_deck)
        self._current_table = Image.open(path_table)
        self._token = Image.open(path_token)
        self._backside = Image.open(path_backside)
        self._trickbackside = Image.open(path_trickbackside)
        self._size = (1700*self.width_ratio, 854*self.height_ratio)
        self.card_box = (73-15,96-15)
        
    @property
    def backside(self):
        res = self._backside
        return res.resize((int(res.size[0]*self.width_ratio)//2, int(res.size[1]*self.height_ratio)//2))
    
    @property
    def trickbackside(self):
        res = self._trickbackside
        return res.resize((int(res.size[0]*self.width_ratio)//2, int(res.size[1]*self.height_ratio)//2))
        
    @property
    def current_table(self):
        return self._current_table.resize(( \
                    int(self._current_table.size[0]*self.width_ratio), int(self._current_table.size[1]*self.height_ratio)))
            
    def table_position(self, offset=0):
        offset = 18*offset
        res = (int(self._size[0]*2/5)+offset, int(self._size[1]/2))
        return res
            
    @property
    def token(self):
        return self._token.resize((int(self._token.size[0]*self.width_ratio/8), int(self._token.size[1]*self.height_ratio/8)))
        
    def make_card(self, suit, val_in):
        H = {'0':'spades', '1':'clubs', '2':'diamonds', '3':'hearts'}
        if type(suit)!=str:
            suit = H[str(suit)]
        suit_num = self.K[suit]
        val = (val_in % 13)+1
        card = self.deck.crop((15+69*val,15+90*suit_num,73+69*val,96+90*suit_num))
        card = card.resize((card.size[0]//2, card.size[1]//2))
        card = card.resize((int(card.size[0]*self.width_ratio), int(card.size[1]*self.height_ratio)))
        return card
    
    def get_player_positions(self, offset=0, player=0):
        offset = 30*offset
        if self.M ==3:
            positions = {'player0':[700,150], 'player1': [900,550], 'player2': [220,500]}
        if self.M ==4:
            positions = {'player0':[350,180], 'player1': [850,180], 'player2': [300,500], 'player3': [850, 500]}
        if self.M ==5:
            positions = {'player0':[190,320], 'player1': [380,170], 'player2': [850,190], 'player3': [900, 580], 'player4': [360, 580]}
        for i in range(self.M):
            key = 'player%s'%str(i)
            positions[key][0] = int((positions[key][0] + offset)*self.width_ratio) 
            positions[key][1] = int(positions[key][1]*self.height_ratio) 
        return positions['player%s'%player]
    
    def get_token_position(self, turn):
        pos = self.get_player_positions(-1,turn)
        return (pos[0] -int(10*self.width_ratio), pos[1])
    
    def get_trick_position(self, offset, turn):
        pos = self.get_player_positions(0,turn)
        return (pos[0]+30*offset, pos[1]+int(60*self.width_ratio))
    
    