# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:11:18 2022

@author: Nagatorinne
"""
import pygame

class video_initial:
    def __init__(self, width=1200, height=800):
        self.window_size=(width,height)
        
    def first_screen(self):
        window_size = self.window_size
        pygame.init()

        screen = pygame.display.set_mode(window_size)
        bg = pygame.image.load('table.png')
        
        running = True
        color_active = pygame.Color('lightskyblue3')
        color_passive = pygame.Color('chartreuse4')
        color = color_passive
        num_M = ''
        num_n = ''
        timing=0
        base_font = pygame.font.Font(None, 32)
        active_1=False
        active_2=False
        start_game= False
        invalid_bool = False
        while running:
            for event in pygame.event.get():
                pygame.event.get()
                screen.fill((0,0,0))
                screen.blit(bg,(0,0))
                input_rect_1 = pygame.Rect(200, 200, 200, 32)
                input_rect_2 = pygame.Rect(200, 400, 200, 32)
                input_rect_3 = pygame.Rect(200, 500, 200, 32)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_rect_1.collidepoint(event.pos):
                        active_1 = True
                    else:
                        active_1 = False
                    if input_rect_2.collidepoint(event.pos):
                        active_2 = True
                    else:
                        active_2 = False
                if event.type == pygame.KEYDOWN:
                    if active_1:
                        if event.key == pygame.K_BACKSPACE:
                            num_M = num_M[:-1]
                        else:
                            num_M += event.unicode
                    if active_2:
                        if event.key == pygame.K_BACKSPACE:
                            num_n = num_n[:-1]
                        else:
                            num_n += event.unicode
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_rect_3.collidepoint(event.pos):
                        start_game = True
            if active_1 or active_2:
                color = color_active
            else:
                color = color_passive
            pygame.draw.rect(screen, color, input_rect_1)
            pygame.draw.rect(screen, color, input_rect_2)
            pygame.draw.rect(screen, 'black', input_rect_3)
            text_surface_start = base_font.render('Start', True, (255, 255, 255))
            screen.blit(text_surface_start, (input_rect_3.x+5, input_rect_3.y+5))
            if num_M =='':
                text_surface = base_font.render('number of players', True, (255, 255, 255))
                screen.blit(text_surface, (input_rect_1.x+5, input_rect_1.y+5))
            else:
                text_surface = base_font.render(num_M, True, (255, 255, 255))
                screen.blit(text_surface, (input_rect_1.x+5, input_rect_1.y+5))
            if num_n =='':
                text_surface = base_font.render('number of cards', True, (255, 255, 255))
                screen.blit(text_surface, (input_rect_2.x+5, input_rect_2.y+5))
            else:
                text_surface = base_font.render(num_n, True, (255, 255, 255))
                screen.blit(text_surface, (input_rect_2.x+5, input_rect_2.y+5))
            pygame.display.flip()
            if invalid_bool:
                rect_4 = pygame.Rect(400, 200, 200, 32)
                pygame.draw.rect(screen, 'black', rect_4)
                text_surface = base_font.render('Invalid', True, (255, 255, 255))
                screen.blit(text_surface, (rect_4.x+5, input_rect_1.y+5))
                pygame.display.flip()
            if start_game:
                try:
                    n = int(num_n)
                    m = int(num_M)
                    pygame.quit()
                    return n,m
                except:
                    invalid_bool = True
                    start_game=False