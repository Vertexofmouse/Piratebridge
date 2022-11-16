# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:10:17 2022

@author: Nagatorinne
"""
from gameclass import game, Player, dec_hand, dec_table # Might need node as well
from Imagehandling import table_images
from videoinitial import video_initial
import pygame
import numpy as np
                    


class play_game(game):
    def __init__(self, m, n, size, players, prints=False):
        super().__init__(n, m, players, prints=prints)
        self.size = size
        self.handler = table_images(m, n, window_size=size)
        self.rects = [None for i in range(52)]
        self.scoretext = ''
        
    def pilImageToSurface(self, image):
        return pygame.image.fromstring(image.tobytes(), image.size, image.mode).convert()
    
    def initialize(self, player=0):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        self.bg = self.pilImageToSurface(self.handler.current_table)
        #pygame.event.get()
        self.screen.fill((0,0,0))
        self.update_window()
        
    def generate_guesses(self):
        for i in range(self.M):
            if self.players[i].strat ==1:
                self.guesses[i] = self.players[i].trick_guesser_1(self.hands[i], self.start, i)
                self.update_window(0)
        for i in range(self.M):
            if self.players[i].strat=='frontend_input':
                self.guesses[i] = self.guess_screen(i)
            else:
                t = np.random.choice(np.array([1,-1], dtype=int))
                guess_0 = np.copy(self.guesses[0])
                if guess_0==0:
                    t= np.random.choice(np.array([1,2], dtype=int))
                elif guess_0==n:
                    t= np.random.choice(np.array([-2,-1], dtype=int))
                else:
                    self.guesses[i] = guess_0 + t
        
    def guess_screen(self, player=0):
        running = True
        color_active = pygame.Color('lightskyblue3')
        color_passive = pygame.Color('chartreuse4')
        color = color_passive
        guess = ''
        base_font = pygame.font.Font(None, 32)
        active=False
        ready = False
        invalid_bool = False
        self.update_window(player)
        while running:
            for event in pygame.event.get():
                pygame.event.get()
                pos = self.handler.table_position()
                input_rect_1 = pygame.Rect(pos[0], pos[1], 100, 32)
                input_rect_2 = pygame.Rect(pos[0], pos[1]+32, 100, 32)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_rect_1.collidepoint(event.pos):
                        active = True
                    else:
                        active = False
                if event.type == pygame.KEYDOWN:
                    if active:
                        if event.key == pygame.K_BACKSPACE:
                            guess = guess[:-1]
                        elif event.key ==pygame.K_RETURN:
                            ready= True
                        else:
                            guess += event.unicode
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_rect_2.collidepoint(event.pos):
                        ready = True
            if active:
                color = color_active
            else:
                color = color_passive
            pygame.draw.rect(self.screen, color, input_rect_1)
            pygame.draw.rect(self.screen, 'black', input_rect_2)
            text_surface_start = base_font.render('Ready', True, (255, 255, 255))
            self.screen.blit(text_surface_start, (input_rect_2.x+5, input_rect_2.y+5))
            if guess =='':
                text_surface = base_font.render('guess', True, (255, 255, 255))
                self.screen.blit(text_surface, (input_rect_1.x+5, input_rect_1.y+5))
            else:
                text_surface = base_font.render(guess, True, (255, 255, 255))
                self.screen.blit(text_surface, (input_rect_1.x+5, input_rect_1.y+5))
            pygame.display.flip()
            if invalid_bool:
                rect = pygame.Rect(pos[0]+100, pos[1]+32, 100, 32)
                pygame.draw.rect(self.screen, 'black', rect)
                text_surface = base_font.render('Invalid', True, (255, 255, 255))
                self.screen.blit(text_surface, (rect.x+5, rect.y+5))
                pygame.display.flip()
            if ready:
                try:
                    guess = int(guess)
                    return guess
                except:
                    invalid_bool = True
                    ready=False
                
    def update_window(self, player=0):
        self.screen.blit(self.bg,(0,0))
        
        tokenpos = self.handler.get_token_position(self.turn)
        token = self.pilImageToSurface(self.handler.token)
        self.screen.blit(token, tokenpos)
        pygame.display.flip()
        
        offsets = np.zeros(self.M, dtype=int)
        for i in range(self.M):
            
            for j in range(52):
                if self.hands[i,j]==1:
                    position = self.handler.get_player_positions(offsets[i], i)
                    offsets[i] +=1
                    if i!= player:
                        card = self.pilImageToSurface(self.handler.backside)
                    else:
                        card = self.handler.make_card(j//13, j%13)
                        card = self.pilImageToSurface(card)
                        rect = card.get_rect()
                        rect.update(position, rect.size)
                        self.rects[j] = rect
                    self.screen.blit(card, position)
                    pygame.display.flip() 
        
        scorerect = pygame.Rect(0, 0, self.size[0], 32)
        base_font = pygame.font.Font(None,25)
        
        text_surface = base_font.render(self.scoretext, True, (0, 0, 0))
        self.screen.blit(text_surface, (scorerect.x+5, scorerect.y+5))
        pygame.display.flip()
        
        for i in range(self.M):
            offset = 0
            for k in range(self.tricks[i]):
                card = self.pilImageToSurface(self.handler.trickbackside)
                position = self.handler.get_trick_position(offset, i)
                offset += 1
                self.screen.blit(card, position)
                pygame.display.flip()
            
            position_s = self.handler.get_player_positions(0, i)        
            guessrect = pygame.Rect(position_s[0], position_s[1]+50, 100, 32)
            base_font = pygame.font.Font(None,23)
            text_surface = base_font.render('player %s guess: '%i+str(self.guesses[i]), True, (0, 0, 0))
            self.screen.blit(text_surface, (guessrect.x+5, guessrect.y+5))
        
        offset = 0
        for k in range(self.start, self.start+self.M):
            h = self.table[k %self.M]
            if h!=-2:
                position = self.handler.table_position(offset)
                offset +=1
                card = self.handler.make_card(h//13, h%13)
                card = self.pilImageToSurface(card)
                self.screen.blit(card, position)
                pygame.display.flip()
                
    
    def generate_move(self, player=0):
        running = True
        active = False
        self.update_window(player)
        while running:
            for event in pygame.event.get():
                
                    
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u:
                        self.update_window(player)
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for j in range(52):
                        if self.rects[j]!=None:
                            if self.rects[j].collidepoint(event.pos):
                                l = j
                                active = True
            if active:
                legal = self.players[player].check_legal(l, self.hands[player], self.table, self.start)
                if legal:
                    running = False
            
        self.hand_sizes[player] -=1
        self.hands[player,l]=0
        self.table[player] = l
        self.burned[l]= 1
        if l//13 != self.table[self.start]//13:
            self.suits_up[player,int(l//13)] = 1
        self.num_moves += 1
    
        
    def play_round(self):
        winner = self.start
        while self.num_moves < self.M:
            move_start = pygame.time.get_ticks()
            if self.prints:
                print('hand: ', dec_hand(self.hands[self.turn]))
            if self.players[self.turn].strat =='frontend_input':
                self.generate_move(self.turn)
            else:
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
            
            time = pygame.time.get_ticks()
            while time - move_start <=1000:
                if int(time - move_start) %100 ==0:
                    self.update_window(0)
                time = pygame.time.get_ticks()
            
            
            
            
            self.turn = (self.turn+1) % self.M
        self.tricks[winner] += 1
        pygame.time.wait(3000)
        self.update_window(0)
        
        
        
        if self.prints:
            print('winner is: ', winner)
        return winner
    
    def play_hand(self, scores):
        if self.prints:
            print('start is in game: ', self.start)
        self.initialize()
        self.generate_guesses()
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
            self.scores[i] = -2*g_t \
                + (2*int(self.tricks[i]) \
                + 8*(self.tricks[i]!=0) \
                + (5 + 15*(self.n>=7))*(self.tricks[i]==0))*(g_t==0)
        
        self.scoretext = ''
        for k in range(self.M):
            self.scoretext += 'player%s score: '%k + str(scores[k]+self.scores[k]) + ',    '
        return self.scoretext
        
    

class game:
    def __init__(self, size=(1200, 800), strats=['frontend_input',1,1,1,1]):
        self.players = list()
        self.size = size
        self.strats = strats
        
    def run(self):
        initial = video_initial(self.size[0], self.size[1])
        useless, m = initial.first_screen()
        cumscores = np.zeros(m)
        scoretext = 'First round'
        V = play_game(m, 1, self.size, self.players, prints=True)
        for n in range(8,10):
            for i in range(m):
                p = Player(n, m, strat= self.strats[i])
                self.players.append(p)
            V.__init__(m, n, self.size, self.players, prints=True)
            V.scoretext = scoretext
            V.start = n % m
            V.turn = n % m
            hands = np.zeros((m,52), dtype=int)
            usable_indices = np.array(list(range(52)))
            np.random.shuffle(usable_indices)
            for p in range(m):
                hand_p = usable_indices[n*p:n*(p+1)]
                hands[p, hand_p] = 1
            V.hands=hands
            scoretext = V.play_hand(cumscores)
            cumscores = cumscores + V.scores
            
G = game()
G.run()







    
