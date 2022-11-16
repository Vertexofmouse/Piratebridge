# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 02:04:36 2022

@author: Nagatorinne
"""

from gameclass import Player
from gameclass import game
from gameclass import dec_hand
import numpy as np



def Play(n, M, its, strats=None, start = None):
    if strats==None:
        strats = [1]+[0]*(M-1)
    players = list()
    for i in range(M):
        p = Player(n, M, strat= strats[i])
        players.append(p)
    G = game(n, M, players, prints=True)
    scores = np.zeros((its, M))
    for k in range(its):
        G.__init__(n, M, players, prints =True)
        if start==None:
            G.start = k % M
            G.turn = k% M
        else:
            G.start=start
            G.turn=start
        hands = np.zeros((M,52), dtype=int)
        usable_indices = np.array(list(range(52)))
        np.random.shuffle(usable_indices)
        for p in range(M):
            hand_p = usable_indices[n*p:n*(p+1)]
            hands[p, hand_p] = 1
        G.hands=hands
        #print('hand of player_0: ', G.hands[0])
        for i in range(M):
            if players[i].strat ==1:
                G.guesses[i] = players[i].trick_guesser_1(G.hands[i], G.start, i)
            elif players[i].strat=='input':
                G.guesses[i] = players[i].trick_input(dec_hand(G.hands[i]), G.start, i)
            else:
                t = np.random.choice(np.array([1,-1], dtype=int))
                guess_0 = np.copy(G.guesses[0])
                if guess_0==0:
                    t= np.random.choice(np.array([1,2], dtype=int))
                elif guess_0==n:
                    t= np.random.choice(np.array([-2,-1], dtype=int))
                else:
                    G.guesses[i] = guess_0 + t
            
        print('guesses: ', G.guesses, 'hand: ', dec_hand(G.hands[0]))
        game_scores = G.play_hand()[1]
        for j in range(M):
            scores[k, j] = game_scores[j]
        print('mean_scores: ', np.sum(scores, axis=0))
    in_data = players[0].strat_1_input_history
    out_data = players[0].strat_1_output_history
    score_data = scores[:,0]
    return in_data, out_data, score_data

def Play_entire(M, its_per_hand, strats=None):
    data_in = list()
    data_out = list()
    data_score = list()
    for n in range(1,10):
        inp, outp, score = Play(n, M, its_per_hand, strats=strats, start = n % M)
        data_in.extend(inp)
        data_out.extend(outp)
        data_score.append(score)
        print('mean_scores of game: ', sum([np.sum(score, axis=0) for score in data_score]))
    return data_in, data_out, data_score
        
data_in, data_out, data_score = Play_entire(5, 1, strats=['input',1,1,1,1])
#data_in, data_out, data_score = Play(6,5, 1000)