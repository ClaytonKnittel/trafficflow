from OpenGL.GL import *
import glfw
import numpy as np

vertexCode = '''
#version 330 core

// update per frame using Camera position
// define a box that is the view frame
uniform vec2 bottomLeft;
uniform vec2 widthHeight;

layout (location = 0) in vec2 position;
layout (location = 1) in vec4 color;

out vec4 p_color;

void main() {
    //gl_Position = vec4(((position - bottomLeft) / widthHeight), 0.0, 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
    p_color = vec4(color);
}
'''

fragmentCode = '''
#version 330 core

in vec4 p_color;

out vec4 color;

void main() {
    color = p_color;
}
'''


class shader:

    def __init__(self, vertexShader=vertexCode, fragmentShader=fragmentCode):
        self.id = glCreateProgram()
        vs = self.add_shader(vertexShader, GL_VERTEX_SHADER)
        fs = self.add_shader(fragmentShader, GL_FRAGMENT_SHADER)

        glAttachShader(self.id, vs)
        glAttachShader(self.id, fs)
        glLinkProgram(self.id)

        if glGetProgramiv(self.id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.id)
            glDeleteProgram(self.id)
            glDeleteShader(vs)
            glDeleteShader(fs)
            raise RuntimeError('Error linking shaders %s' % info)
        glDeleteShader(vs)
        glDeleteShader(fs)

    def __del__(self):
        glDeleteProgram(self.id)

    def add_shader(self, vertexShader, type):
        try:
            shader_id = glCreateShader(type)
            glShaderSource(shader_id, vertexShader)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            raise

    def uniform_loc(self, name):
        return glGetUniformLocation(self.id, name)

    def use(self):
        glUseProgram(self.id)


class glyph:

    PTS_PER_VERTEX = 6
    GL_FLOAT_SIZE = 4

    def __init__(self, shape, color, mode=GL_TRIANGLES):
        self.shape = shape
        self.color = color
        self.mode = mode
        self._init_vobjs()

    def __del__(self):
        glDeleteBuffers(1, self.vbo)
        glDeleteVertexArrays(1, self.vao)

    def _init_vobjs(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, glyph.PTS_PER_VERTEX * glyph.GL_FLOAT_SIZE, 0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, glyph.PTS_PER_VERTEX * glyph.GL_FLOAT_SIZE, 2 * glyph.GL_FLOAT_SIZE)

        self.load_data()

    def gen_data(self):
        data = []
        for pt in self.shape.points:
            data.append(pt[0])
            data.append(pt[1])
            data.append(self.color[0])
            data.append(self.color[1])
            data.append(self.color[2])
            data.append(self.color[3])
        return np.array(data)

    def load_data(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        data = self.gen_data()
        self.size = int(len(data) / glyph.PTS_PER_VERTEX)
        print(data.size, data.itemsize)
        glBufferData(GL_ARRAY_BUFFER, data.size * data.itemsize, data, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glDrawArrays(self.mode, 0, self.size)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


class window:

    def __init__(self, width, height, title=''):
        if not glfw.init():
            glfw.terminate()
            raise RuntimeError('Could not initialize GLFW')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.w = glfw.create_window(width, height, title, None, None)
        self.__dim = glfw.get_framebuffer_size(self.w)
        glfw.make_context_current(self.w)

        self.prog = shader()

        self.framerate = 30
        self.bg_color = (1., 1., 1., 1.)

        glViewport(0, 0, *self.__dim)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def __del__(self):
        del self.prog
        glfw.destroy_window(self.w)
        glfw.terminate()

    def setBGColor(self, *color):
        self.bg_color = color

    def set_framerate(self, framerate):
        self.framerate = framerate

    def begin_draw(self, camera):
        glfw.poll_events()
        glClearColor(*self.bg_color)
        glClear(GL_COLOR_BUFFER_BIT)
        self.prog.use()
        glUniform2f(self.prog.uniform_loc('bottomLeft'), *camera.bottomLeft())
        glUniform2f(self.prog.uniform_loc('widthHeight'), *camera.widthHeight())

    def end_draw(self):
        glfw.swap_buffers(self.w)

    def should_close(self):
        return glfw.window_should_close(self.w)
