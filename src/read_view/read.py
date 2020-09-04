import glfw
from OpenGL.GL import *
import numpy as np
from numpy.core.fromnumeric import trace
from read_file import read_file
from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.arrays import vbo
from OpenGL.GL.shaders import compileShader, compileProgram
if not glfw.init():
    raise Exception("glfw not initialized")

window =  glfw.create_window(1280,720,"OpenGL Window",None,None)


if not window:
    glfw.terminate()
    raise Exception("glf window can't be created")

glfw.set_window_pos(window,400,200)

glfw.make_context_current(window)


vertex_src = """
# version 330
in vec3 a_position;
in vec3 a_color;
out vec3 v_color;
void main()
{
    gl_Position = vec4(a_position, 1.0);
    v_color = a_color;
}
"""

fragment_src = """
# version 330
in vec3 v_color;
out vec4 out_color;
void main()
{
    out_color = vec4(v_color, 1.0);
}
"""


shader = compileProgram(compileShader(vertex_src,GL_VERTEX_SHADER),compileShader(fragment_src,GL_FRAGMENT_SHADER))

glUseProgram(shader)


path = r"C:\Users\samme\Google_Drive\Code_library\MIR_Pipeline_Project\data\benchmark\db\3\m300\m300.off"
vertices, triangle_elements, info = read_file(path)

VBO = glGenBuffers(1)

glBindBuffer(GL_ARRAY_BUFFER,VBO)

glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW)

position = glGetAttribLocation(shader,"a_position")

glEnableVertexAttribArray(position)

glVertexAttribPointer(position,vertices.shape[0],GL_FLOAT,GL_FALSE,0,ctypes.c_void_p(0))








while not glfw.window_should_close(window):
    glfw.poll_events()

  
    #glDrawArrays(GL_TRIANGLES, 0, 3) #This line still works
    glDrawElements(GL_TRIANGLES, info[1], GL_UNSIGNED_INT, None) #This line does work too!


    # glRotatef(2,1,0,0)

    # glDrawElements(GL_TRIANGLES,3,GL_UNSIGNED_INT,None)
    


    glfw.swap_buffers(window)
glfw.terminate()
