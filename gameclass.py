# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 02:18:57 2022

@author: Nagatorinne
"""

import numpy as np
import torch, torchvision
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader



def dec_hand(X): 
    n = np.sum(X)
    indices = np.where(X>0.5)[0]
    out = np.zeros((n,2))
    for i in range(n):
        out[i,0] = indices[i]//13
        out[i,1] = indices[i]%13+1
    return out

def dec_table(table):
    out = np.zeros((table.shape[0], 2))
    for i in range(table.shape[0]):
        if table[i] !=-2:
            out[i,0]= table[i]//13
            out[i,1]= table[i]%13+1 
        else:
            out[i,0]= -2
            out[i,1]= -2 
    return out


class Node:
    def __init__(self, n, M, hand):
        self.n = n
        self.M = M
        self.hand = hand
        self.hand_size = np.sum(self.hand)
        self.parent = None
        self.children = [0]*52
        self.current_card = None
        self.visited = np.zeros(52, dtype=int)
        self.num_visits = np.zeros(52, dtype=int)
        self.utility = np.zeros(52) 
        
    def next_node(self, legal_inds): # Forward and backward MCTS
        if np.sum(self.visited[legal_inds]) ==legal_inds.shape[0]:
            J = np.argmax((self.utility+ 2*np.sqrt(2*np.log(np.sum(1+self.num_visits))/(1+self.num_visits)))[legal_inds])
            I = legal_inds[J]
            self.num_visits[I] += 1
        else:
            J = np.argmin(self.visited[legal_inds])
            I = legal_inds[J]
            self.num_visits[I] += 1
            self.visited[I]=1

        if self.children[I]==0 and legal_inds.shape[0]>0:
            hand_child = np.copy(self.hand)
            hand_child[I] = 0
            child = Node(self.n, self.M, hand_child)
            child.parent = self
            child.current_card = I
            self.children[I]=child
            #print('visited index in node: ', I, 'current card: ', self.current_card)
        else:
            self.children[I].current_card = I
        return I, self.children[I]
            

class Player:
    def __init__(self, n, M, engine=None, strat=0):
        self.n = n
        self.M = M
        self.engine = engine
        self.strat = strat            
        self.suits = np.zeros((M, 52), dtype=int)
        self.current_node = None
        self.strat_1_input_history = list()
        self.strat_1_output_history = list()
        for j in range(4):
            self.suits[j,j*13:(j+1)*13] = 1
        self.card=None
    
    def check_legal(self, card, hand, table, start):
        x = card//13
        if table[start] == -2:
            y = x
        else:
            y = table[start]//13
        check_table = (x==y)
        check_suit = False
        check_hand= hand[card]==1
        for i in range(52):
            check_suit = check_suit or (hand[i]==1 and i//13==y)
        #print('check table: ', check_table, 'check hand: ', check_hand, 'returns: ', check_hand and (check_table or (not check_suit)))
        return check_hand and (check_table or (not check_suit))
    
    def trick_guesser(self, hand, start, this_player_turn):
        table = -2*np.ones(self.M, dtype=int)
        turn = 0
        winner= 0
        tricks = np.zeros(self.M, dtype=int)
        scores = np.zeros(self.M)
        num_moves = 0
        num_rounds = 0
        suits_up = np.zeros((self.M,4), dtype=int)
        burned = np.zeros(52, dtype=int)
        hand_sizes = self.n*np.ones(self.M, dtype=int)
        res = np.zeros(self.n+1)
        for guess in range(self.n+1):
            guesses = ((self.n-guess +1)/(self.M-1))*np.ones(self.M, dtype=int)
            guesses[this_player_turn] = guess
            close = self.strategy_1(10, 100, hand, table, start, turn, winner, tricks, guesses, scores, num_moves, num_rounds, suits_up, burned, hand_sizes, )
            #close += self.strategy_1(10, 100, hand, table, start, turn, winner, tricks, guesses, scores, num_moves, num_rounds, suits_up, burned, hand_sizes)
            res[guess] = np.amax(close[np.where(hand>0.5)[0]])
        print('res: ', res)
        I = np.argmax(res)
        
        return I
    
    def trick_guesser_1(self, hand, start, this_player_turn):
        strats = [1]*self.M
        players = list()
        for i in range(self.M):
            p = Player(self.n, self.M, strat= strats[i])
            players.append(p)
        G = game(self.n, self.M, players)
        G.K = 400
        res = np.zeros(self.n+1)
        for guess in range(self.n+1):
            used_cards = np.copy(hand)
            hands = np.zeros((self.M, 52), dtype=int)
            for p in range(self.M):
                if p == this_player_turn:
                    hands[p]= hand
                else:
                    usable_indices = np.where(used_cards<0.5)[0]
                    np.random.shuffle(usable_indices)
                    hand_p = usable_indices[0:self.n]
                    hands[p, hand_p] = 1
                    used_cards += np.copy(hands[p])
                    #print('used', used_cards, 'hand for player ', p, ' is ', hands[p])
                    
            L = [dec_hand(hand) for hand in hands]
            #print('hands: ', L)
            guesses = ((self.n-guess +1)/(self.M-1))*np.ones(self.M, dtype=int)
            guesses[this_player_turn] = guess
            #print('game reset')
            G.__init__(self.n, self.M, players)
            G.start = start
            G.hands = hands
            G.K=400
            res[guess] = G.play_hand()[0][this_player_turn] 
        I = np.argmin(res)
        
        return I
    
    def trick_input(self, hand, start, turn):
        print('0 is spades, 1 is clubs, 2 is diamonds, 3 is hearts')
        print('hand is ', hand, 'start is ', start, 'your turn is ', turn)
        J = input('tricks out of %s cards '%self.n)
        return int(J)
    
    
    def strategy(self, node=None, card=None, **kwargs):
        #print('strat: ', self.strat)
        strat = self.strat
        if strat==0:
            hand = kwargs['hand']
            table= kwargs['table']
            start= kwargs['start']
            return self.strategy_0(hand, table, start)
        elif strat==1:
            K = kwargs['K']
            hand = kwargs['hand']
            table= kwargs['table']
            start= kwargs['start']
            turn = kwargs['turn']
            winner = kwargs['winner']
            tricks = kwargs['tricks']
            guesses = kwargs['guesses']
            scores = kwargs['scores']
            num_moves=kwargs['num_moves']
            num_rounds=kwargs['num_rounds']
            suits_up = kwargs['suits_up']
            burned = kwargs['burned']
            hand_sizes = kwargs['hand_sizes']
            hand_indices = np.where(hand>0.5)[0]
            legal_vector = np.vectorize(lambda x: self.check_legal(x, hand, table, start))(hand_indices)
            legal_indices = hand_indices[np.where(legal_vector)[0]]
            average_scores = self.strategy_1(40, K//40+1, hand, table, start, turn, winner, tricks, guesses, scores, num_moves, num_rounds, suits_up, burned, hand_sizes)
            j = np.argmax(average_scores[legal_indices])
            l = legal_indices[j]
            legality = self.check_legal(l, hand, table, start)
            #print('strategy_1 legal? ', legality, 'card is: ', [l//13, l%13 +1], 'hand is: ', dec_hand(hand), 'turn: ', turn)
            self.strat_1_output_history.append(l)
            return l
        elif strat=='input':
            hand = kwargs['hand']
            table= kwargs['table']
            start= kwargs['start']
            turn = kwargs['turn']
            b = False
            while not b:
                if self.card==None:
                    print('0 is spades, 1 is clubs, 2 is diamonds, 3 is hearts')
                    print('hand is ', dec_hand(hand), 'start is ', start, 'your turn is ', turn)
                    suit = input('suit ')
                    value = input('value ')
                    suit_dict = {'spades': 0, 'clubs': 1, 'diamonds': 2, 'hearts': 3}
                    l = 13*suit_dict[suit] + int(value)-1
                b = self.check_legal(l, hand, table, start)
                print('legal move? ', b, 'card is ', l)
            return l
        else:
            hand = kwargs['hand']
            table= kwargs['table']
            start= kwargs['start']
            return self.node_strat(hand, table, start)
    
    def strategy_0(self, hand, table, start):
        hand_indices = np.where(hand>0.5)[0]
        np.random.shuffle(hand_indices)
        k = 0
        while not self.check_legal(hand_indices[k], hand, table, start):
            k = k+1
        l = hand_indices[k]
        return l
    
    def node_strat(self, hand, table, start): #Forward 1 step
        hand_indices = np.where(hand>0.5)[0]
        legal_vector = np.vectorize(lambda x: self.check_legal(x, hand, table, start))(hand_indices)
        legal_indices = hand_indices[np.where(legal_vector)[0]]
        #print('hand: ', hand, 'legal_indices: ', legal_indices)
        #print('hand in node_strat: ', dec_hand(hand), 'current node utility: ', self.current_node.utility)
        I, child = self.current_node.next_node(legal_indices)
        #print('hand_indices: ', hand_indices, 'visited: ', self.current_node.visited[hand_indices])
        self.current_node = child
        return I
    
    def node_update_rec(self, score, node): #Backward
        if node.parent != None:
            #print('current_ node: ', node.hand_size)
            I = node.current_card 
            #node.parent.utility[I] += (score +5)/20 # utility is average score
            node.parent.utility[I] += (score<= 0.5) -1*(score>0.5) # utility is wether the guess was correct or not 
            self.node_update_rec(score, node.parent)
    def node_update(self, score, root): #Backward
        self.node_update_rec(score, self.current_node)
        self.current_node = root
            
    
    def strategy_1(self, M, K, hand, table, start, turn, winner, tricks, guesses, scores, num_moves, num_rounds, suits_up, burned, hand_sizes, player_strats=None):
        self.strat_1_input_history.append([np.copy(hand), np.copy(table), start, turn, np.copy(tricks), np.copy(guesses), np.copy(suits_up), np.copy(burned)])
        #print('strategy_1 called')
        average_scores = np.zeros(52)
        root = Node(self.n, self.M, hand)
        for i in range(M): #determinizations
            used_cards = hand+burned
            hands = np.zeros((self.M, 52), dtype=int)
            for p in range(self.M):
                if p == turn:
                    hands[p]= hand
                else:
                    unusable_cards = used_cards + sum([suits_up[p,s]*self.suits[s] for s in [0,1,2,3]])
                    usable_indices = np.where(unusable_cards<0.5)[0]
                    np.random.shuffle(usable_indices)
                    hand_p = usable_indices[0:hand_sizes[p]]
                    hands[p, hand_p] = 1
                    used_cards += np.copy(hands[p])
                
            parameters = [self.n, self.M, np.copy(hands), np.copy(table), start, turn, winner, np.copy(tricks), np.copy(guesses), np.copy(scores), num_moves, num_rounds, np.copy(suits_up), np.copy(burned), np.copy(hand_sizes)]
            players = list(range(self.M))
            if player_strats==None:
                for i in range(self.M):
                    if i == turn:
                        self.strat=3
                        players[i] =self
                    else:
                        players[i] = Player(self.n, self.M)
            else:
                for i in range(self.M):
                    if i == turn:
                        self.strat=3
                        players[i] =self
                    else:
                        players[i] = Player(self.n, self.M)
                        players[i].strats= player_strats[i]
            sim_1 = game(self.n, self.M, players)
            sim_1.set_game_state(parameters)
            #hand_indices = np.where(sim_1.hands[turn]>0.5)[0]
            #legal_vector = np.vectorize(lambda x: self.check_legal(x, hands[turn], table, start))(hand_indices)
            #legal_indices = hand_indices[np.where(legal_vector)[0]]
            #root = Node(self.n, self.M, hand)
            players[turn].current_node = root
            for k in range(K): # MCTS
                parameters = [self.n, self.M, np.copy(hands), np.copy(table), start, turn, winner, np.copy(tricks), np.copy(guesses), np.copy(scores), num_moves, num_rounds, np.copy(suits_up), np.copy(burned), np.copy(hand_sizes)]
                #print('iteration; ', k)
                sim_1.set_game_state(parameters)
                #score = sim_1.play_hand()[1]
                score = sim_1.play_hand()[0]
                this_playerscore = score[turn]
                self.node_update(this_playerscore, root)
                #print('root_utility: ', root.utility)
            average_scores += root.utility
        #print('average scores: ', average_scores)
        self.strat=1
        return average_scores


class game:
    def __init__(self, n, M, Players, prints=False):
        self.n = n
        self.M = M
        self.prints = prints
        self.hands = np.zeros((M,52), dtype=int)
        self.table = -2*np.ones(M, dtype=int)
        self.start = 0
        self.turn = 0
        self.winner= 0
        self.tricks = np.zeros(self.M, dtype=int)
        self.guesses = np.zeros(self.M, dtype=int)
        self.scores = np.zeros(self.M)
        self.num_moves = 0
        self.num_rounds = 0
        self.suits_up = np.zeros((M,4), dtype=int)
        self.burned = np.zeros(52, dtype=int)
        self.hand_sizes = n*np.ones(M, dtype=int)
        self.players = Players
        self.K = 4000
        
    def set_game_state(self, parameters): #parameters = [n, M, hands, table, start, turn, winner, tricks, guesses, scores, num_moves, num_rounds, suits_up, burned, hand_sizes]
        self.n = parameters[0]
        self.M = parameters[1]
        self.hands = parameters[2]
        self.table = parameters[3]
        self.start = parameters[4]
        self.turn = parameters[5]
        self.winner= parameters[6]
        self.tricks = parameters[7]
        self.guesses = parameters[8]
        self.scores = parameters[9]
        self.num_moves = parameters[10]
        self.num_rounds = parameters[11]
        self.suits_up = parameters[12]
        self.burned = parameters[13]
        self.hand_sizes = parameters[14]
            
    def move(self, i):
        #print('strat in game: ', self.players[i].strat, 'for player: ', i)
        l = self.players[i].strategy(K=self.K, hand=self.hands[i], table=self.table, start=self.start, turn=self.turn, winner=self.winner, \
                                     tricks=self.tricks, guesses=self.guesses, scores=self.scores, \
                                         num_moves=self.num_moves, num_rounds=self.num_rounds, suits_up=self.suits_up, burned=self.burned, \
                                             hand_sizes=self.hand_sizes)
        self.hand_sizes[i] -=1
        self.hands[i,l]=0
        self.table[i] = l
        self.burned[l]= 1
        if l//13 != self.table[self.start]//13:
            self.suits_up[i,int(l//13)] = 1
        self.num_moves += 1
        
    
    def play_round(self):
        winner = self.start
        while self.num_moves < self.M:
            #if self.prints:
                #print('hand: ', dec_hand(self.hands[self.turn]))
            self.move(self.turn)
            if self.prints:
                print('table: ', dec_table(self.table))
            
            suit = self.table[self.turn]//13
            suit_winner = self.table[winner]//13
            number = self.table[self.turn]%13
            number_winner = self.table[winner]%13
            if  (suit == suit_winner and number > number_winner) or \
            (suit==0 and suit != suit_winner):
                winner = self.turn
            self.turn = (self.turn+1) % self.M
        self.tricks[winner] += 1
        if self.prints:
            print('winner is: ', winner)
        return winner
    
    def play_hand(self):
        if self.prints:
            print('start is in game: ', self.start)
        while self.num_rounds < self.n:
            winner = self.play_round()
            if self.prints:
                print('table: ', dec_table(self.table))
            self.turn = winner
            self.winner = winner
            self.start = winner
            self.num_rounds += 1
            self.num_moves = 0
            self.table = -2*np.ones(self.M, dtype=int)
        for i in range(self.M):
            g_t = int(np.abs(self.guesses[i] - self.tricks[i]))
            self.scores[i] = -2*g_t + (2*int(self.tricks[i]) + 8*(self.tricks[i]!=0) + (5 + 15*(self.n>=7))*(self.tricks[i]==0))*(g_t==0)
        return (self.tricks- self.guesses)**2, self.scores
                    
                    
    

        
        
        
                    
                    
                    
                    
                    
                    
                    
                    
                