from cmath import pi, sqrt
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from glfw import PRESS
import numpy as np
import pyrr



class input_handler:
    def __init__(self,window):
        self.mode_list = [0,1]
        self.mode = 0
        self.window = window
        self.eye= np.array([0,0,3],dtype=np.float32)
        self.target = np.array([0,0,0])
        self.up = np.array([0,1,0])
        
        #for scroll zoom
        self.current_distance = 1
        self.zoom_rate = 1/5
        
        #for rotation
        self.rotation_list = np.array([0,0,0],dtype=np.float32)
        self.rotations_per_screen_hor = 0.2
        self.rotations_per_screen_vert = 0.5
        
        self.right_key_pressed = False
    
    def scroll_handler(self,window,xoffset,yoffset):
        self.eye[2] -=  yoffset * self.zoom_rate
        self.eye[2] = max(self.eye[2],0)
   
    
    def mouse_handler(self,window,button,action,mods):
    
        if button==glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.start_cursor_position = glfw.get_cursor_pos(window)
            self.right_key_pressed=True
        if action == glfw.RELEASE:
            self.right_key_pressed= False
        
    def keyboard_handler(self,window,key,scancode,action,mods):
        
        if key == glfw.KEY_SPACE and action ==glfw.PRESS:
            self.mode +=1
            self.mode = self.mode % len(self.mode_list)
            print("space>")

        

