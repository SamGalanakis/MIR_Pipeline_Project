from cmath import pi
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from glfw import PRESS
import numpy as np
import pyrr
from pathlib import Path
from read_file import read_file
from input_handler import input_handler
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('f', type=str, help='file path')

args = parser.parse_args()

path =  Path(args.f)


# path = str(Path("data/test_ply.ply"))





vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec3 v_color;
void main()
{
   gl_Position = projection * view * model * vec4(a_position, 1.0);
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

def window_resize(window, width, height):
    glViewport(0, 0, width, height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
   # projection = pyrr.matrix44.create_orthogonal_projection_matrix(0,1280,0,720,-1000,1000)
    window_height = height
    window_width = width
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

input_handler = input_handler(window)
# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 400, 200)

# set the callback function for window resize

glfw.set_window_size_callback(window, window_resize)
# make the context current

glfw.make_context_current(window)



vertices, indices, info = read_file(path)

as_points = vertices.reshape(-1,3)

mean_point = as_points.mean(axis=0)

max_x, max_y,max_z = as_points.max(axis=0)
min_x, min_y , min_z = as_points.min(axis=0)

middle_point = np.array([min_x + (max_x-min_x)/2 , min_y + (max_y-min_y)/2 , min_z + (max_z-min_z)/2] )









shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))



# Vertex Buffer Object
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Element Buffer Object
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(12))

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)

glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
glEnable(GL_DEPTH_TEST)

## Shader matrices 
model_loc = glGetUniformLocation(shader, "model")

proj_loc = glGetUniformLocation(shader, "projection")

view_loc = glGetUniformLocation(shader,"view")

window_height = glfw.get_window_size(window)[1]
window_width = glfw.get_window_size(window)[0]
projection = pyrr.matrix44.create_perspective_projection_matrix(fovy=45,aspect=window_width/window_height,near=0.1,far=100)
#projection = pyrr.matrix44.create_orthogonal_projection_matrix(0,1280,0,720,-1000,1000)

scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([1]*3))

 

# eye pos , target, up
view = pyrr.matrix44.create_look_at(pyrr.Vector3([0,0,3]),pyrr.Vector3([0,0,0]),pyrr.Vector3([0,1,0]))
proj_matrix = glGetUniformLocation(shader,"projection")


initial_offset = middle_point 
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3(-initial_offset))


print(f"Initial offset: {initial_offset}")


## Input

move = 0
rotation = pyrr.matrix44.create_from_axis_rotation(np.array([0,1,0]),0)
def keyboard_input_manager(window,key,scancode,action,mods):
    global move
    global rotation
    if action == glfw.PRESS:
        pass
    
    
    

glfw.set_key_callback(window,input_handler.keyboard_handler)
glfw.set_scroll_callback(window,input_handler.scroll_handler)
glfw.set_mouse_button_callback(window,input_handler.mouse_handler)




previous_displacement = np.zeros(2)
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


    if input_handler.right_key_pressed:
        current_cursor_position = glfw.get_cursor_pos(window)
        cursor_displacement = np.array(current_cursor_position ) - np.array(input_handler.start_cursor_position) - previous_displacement
        input_handler.rotation_list[0] += input_handler.rotations_per_screen_hor * cursor_displacement[0]/window_width
        input_handler.rotation_list[1] += input_handler.rotations_per_screen_vert * cursor_displacement[1]/window_height
        previous_displacement = cursor_displacement
        
    

    rot_x = pyrr.Matrix44.from_x_rotation(input_handler.rotation_list[1])

    rot_y = pyrr.Matrix44.from_y_rotation(input_handler.rotation_list[0])

    rotation = pyrr.matrix44.multiply(rot_x, rot_y)

  
   
    view = pyrr.matrix44.create_look_at(pyrr.Vector3(input_handler.eye),pyrr.Vector3(input_handler.target),pyrr.Vector3(input_handler.up))
 
  
    model= pyrr.matrix44.multiply(scale,translation)
    model= pyrr.matrix44.multiply(model,rotation)
    glUniformMatrix4fv(view_loc,1,GL_FALSE,view)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glUniformMatrix4fv(proj_matrix, 1, GL_FALSE, projection) 
    

    if input_handler.mode:
        glDrawElements(GL_LINES, len(indices), GL_UNSIGNED_INT, None)
    else:
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)


    

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()