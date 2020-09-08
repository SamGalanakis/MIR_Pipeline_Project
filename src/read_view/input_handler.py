from cmath import pi, sqrt
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from glfw import PRESS
import numpy as np
import pyrr


class input_handler:
    def __init__(self,window):
        self.window = window
        self.eye= np.array([0,0,3])
        self.target = np.array([0,0,0])
        self.up = np.array([0,1,0])
        
        #for scroll zoom
        self.current_distance = 1
        self.zoom_rate = 1/5
        

    
    def scroll_handler(self,window,xoffset,yoffset):
        self.eye[2] = self.eye[2] +  yoffset * self.zoom_rate
        print(yoffset * self.zoom_rate)
        print(yoffset)
        print(self.eye)

