from cmath import pi
import glfw
import platform
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from glfw import PRESS
import numpy as np
import pyrr
from file_reader import FileReader
from input_handler import InputHandler


class ModelViwer:
    def __init__(self):

        self.rotation_list = [0, 0, 0]
        self.view_list = [0, 0, 3]

        self.right_key_pressed = False
        self.start_cursor_position = (0, 0)

        self.rotations_per_screen_vert = 0.1
        self.rotations_per_screen_hor = 0.5
        self.previous_displacement = np.array([0, 0])

        self.cursor_displacement = None

        self.vertex_src = """
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

        self.fragment_src = """
        # version 330
        in vec3 v_color;
        out vec4 out_color;
        void main()
        {
            out_color = vec4(v_color, 1.0);
        }
        """

    def window_resize(self, window, width, height):
        glViewport(0, 0, width, height)
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, width / height, 0.1, 100)
        window_height = glfw.get_window_size(window)[1]
        window_width = glfw.get_window_size(window)[0]
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    def keyboard_input_manager(self, window, key, scancode, action, mods):
        translation_val = 0.3
        if key == glfw.KEY_LEFT and action == glfw.PRESS:
            self.view_list[0] -= translation_val
        elif key == glfw.KEY_RIGHT and action == glfw.PRESS:
            self.view_list[0] += translation_val
        elif key == glfw.KEY_UP and action == glfw.PRESS:
            self.view_list[1] += translation_val
        elif key == glfw.KEY_DOWN and action == glfw.PRESS:
            self.view_list[1] -= translation_val

    def mouse_input_manager(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.start_cursor_position = glfw.get_cursor_pos(window)
                self.right_key_pressed = True
            if action == glfw.RELEASE:
                self.right_key_pressed = False

    def rotate(self, x, y, z):
        """x,y,z should be degrees"""
        x *= pi/180
        y *= pi/180
        z *= pi/180

        rot_x = pyrr.Matrix44.from_x_rotation(x)

        rot_y = pyrr.Matrix44.from_y_rotation(y)

        rot_z = pyrr.Matrix44.from_z_rotation(z)

    def process(self, path):
        # initializing glfw library
        if not glfw.init():
            raise Exception("glfw can not be initialized!")

        # creating the window
        window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

        input_handler = InputHandler(window)
        # check if window was created
        if not window:
            glfw.terminate()
            raise Exception("glfw window can not be created!")

        # set window's position
        glfw.set_window_pos(window, 400, 200)

        # set the callback function for window resize

        glfw.set_window_size_callback(window, self.window_resize)
        # make the context current

        glfw.make_context_current(window)

        reader = FileReader(path)
        vertices, indices, info = reader.read()

        as_points = vertices.reshape(-1, 3)

        mean_point = as_points.mean(axis=0)

        max_x, max_y, max_z = as_points.max(axis=0)
        min_x, min_y, min_z = as_points.min(axis=0)

        shader = compileProgram(compileShader(self.vertex_src, GL_VERTEX_SHADER), compileShader(
            self.fragment_src, GL_FRAGMENT_SHADER))

        # Vertex Buffer Object
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes,
                     vertices, GL_STATIC_DRAW)

        # Element Buffer Object
        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     indices.nbytes, indices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(12))

        glUseProgram(shader)
        glClearColor(0, 0.1, 0.1, 1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # Shader matrices
        model_loc = glGetUniformLocation(shader, "model")

        proj_loc = glGetUniformLocation(shader, "projection")

        view_loc = glGetUniformLocation(shader, "view")

        window_height = glfw.get_window_size(window)[1]
        window_width = glfw.get_window_size(window)[0]
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            fovy=45, aspect=window_width/window_height, near=0.1, far=100)
        #projection = pyrr.matrix44.create_orthogonal_projection_matrix(0,1280,0,720,-1000,1000)

        scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([1, 1, 1]))

        # eye pos , target, up
        view = pyrr.matrix44.create_look_at(pyrr.Vector3(
            [0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
        proj_matrix = glGetUniformLocation(shader, "projection")

        translation = pyrr.matrix44.create_from_translation(
            pyrr.Vector3([0, 0, 0]))

        glfw.set_key_callback(window, self.keyboard_input_manager)
        glfw.set_scroll_callback(window, input_handler.scroll_handler)
        glfw.set_mouse_button_callback(window, self.mouse_input_manager)

        while not glfw.window_should_close(window):
            glfw.poll_events()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if self.right_key_pressed:
                current_cursor_position = glfw.get_cursor_pos(window)
                self.cursor_displacement = np.array(
                    current_cursor_position) - np.array(self.start_cursor_position) - self.previous_displacement
                self.rotation_list[0] += self.rotations_per_screen_hor * \
                    self.cursor_displacement[0]/window_width
                self.rotation_list[1] += self.rotations_per_screen_vert * \
                    self.cursor_displacement[1]/window_height
                self.previous_displacement = self.cursor_displacement
                print(self.cursor_displacement)

            rot_x = pyrr.Matrix44.from_x_rotation(self.rotation_list[1])

            rot_y = pyrr.Matrix44.from_y_rotation(self.rotation_list[0])

            rotation = pyrr.matrix44.multiply(rot_x, rot_y)

            current_scale = 1
            scale = pyrr.matrix44.create_from_scale(pyrr.Vector3(
                [current_scale, current_scale, current_scale]))
            view = pyrr.matrix44.create_look_at(pyrr.Vector3(input_handler.eye), pyrr.Vector3(
                input_handler.target), pyrr.Vector3(input_handler.up))
            # view = pyrr.matrix44.create_from_translation(pyrr.Vector3(self.view_list))
            translation_rotation = pyrr.matrix44.multiply(
                rotation, translation)
            model = pyrr.matrix44.multiply(scale, translation_rotation)
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
            glUniformMatrix4fv(proj_matrix, 1, GL_FALSE, projection)
            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

            glfw.swap_buffers(window)

        # terminate glfw, free up allocated resources
        glfw.terminate()
