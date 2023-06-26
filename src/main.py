#!/usr/bin/env python

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image

# ### Inicializando janela
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
altura = 1200
largura = 1600
window = glfw.create_window(largura, altura, "Malhas e Texturas", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;

        varying vec2 out_texture;
        varying vec3 out_fragPos;
        varying vec3 out_normal;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(position);
            out_normal  = normals;
        }
        """


# ### `GLSL` para *Fragment Shader*
fragment_code = """
        uniform vec3 lightPos;
        uniform float ka;
        uniform float kd;

        vec3 lightColor = vec3(1.0, 1.0, 1.0);

        varying vec2 out_texture;
        varying vec3 out_normal;
        varying vec3 out_fragPos;
        uniform sampler2D samplerTexture;
        
        void main(){
            vec3 ambient = ka * lightColor;

            vec3 norm = normalize(out_normal);
            vec3 lightDir = normalize(lightPos - out_fragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = kd * diff * lightColor;

            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient + diffuse), 1.0) * texture;
            gl_FragColor = result;
        }
        """


# ### Requisitando slot para a GPU para nossos programas *Vertex* e *Fragment Shaders*

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)


# ### Associando nosso código-fonte aos slots solicitados

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)


# ### Compilando o *Vertex Shader*
# Se há algum erro em nosso programa *Vertex Shader*, nosso app para por aqui.

# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")


# ### Compilando o *Fragment Shader*
# Se há algum erro em nosso programa *Fragment Shader*, nosso app para por aqui.

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")


# ### Associando os programas compilados ao programa principal

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)


# ### Linkagem do programa

# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)


# ### Preparando dados para enviar à GPU
# Nesse momento, compilamos nossos *Vertex* e *Fragment Shaders* para que a GPU possa processá-los.
# 
# Por outro lado, as informações de vértices geralmente estão na CPU e devem ser transmitidas para a GPU.
# ### Carregando Modelos (vértices e texturas) a partir de Arquivos
# A função abaixo carrega modelos a partir de arquivos no formato *WaveFront*.

def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    normals = []
    texture_coords = []
    faces = []

    material = None

    # abre o arquivo obj para leitura
    for line in open(filename, "r"): ## para cada linha do arquivo .obj
        if line.startswith('#'): continue ## ignora comentarios
        values = line.split() # quebra a linha por espaço
        if not values: continue


        ### recuperando vertices
        if values[0] == 'v':
            vertices.append(values[1:4])

        if values[0] == 'vn':
            normals.append(values[1:4])

        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])

        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            face_normals = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) > 2:
                    face_normals.append(int(w[2]))
                else:
                    face_normals.append(0)
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, face_normals, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces
    model['normals'] = normals

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 10
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    #image_data = np.array(list(img.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)


# ### A lista abaixo armazena todos os vertices carregados dos arquivos
vertices_list = []    
normals_list = []
textures_coord_list = []


# ### Carregamos cada modelo e definimos funções para desenhá-los

modelo = load_model_from_file('../modelos/lake.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo caixa.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id - 1])
print('Processando modelo caixa.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(0,'../texturas/water.jpg')

modelo = load_model_from_file('../modelos/terreno2.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo terreno.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id - 1])
print('Processando modelo terreno.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(1,'../texturas/grama.jpg')

modelo = load_model_from_file('../modelos/snow-cottage.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo medieval-house.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id - 1])
print('Processando modelo medieval-house.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(2,'../texturas/snow-cottage.jpg')

modelo = load_model_from_file('../modelos/boat.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo monstro.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id - 1])
print('Processando modelo monstro.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(3,'../texturas/boat.jpg')

modelo = load_model_from_file('../modelos/sol.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo sol.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    #for normal_id in face[2]:
    #    normals_list.append( modelo['normals'][normal_id - 1])
print('Processando modelo sol.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(4,'../texturas/sun.jpg')

# ### Para enviar nossos dados da CPU para a GPU, precisamos requisitar slots.
# 
# Agora requisitamos dois slots.
# * Um para enviar coordenadas dos vértices.
# * Outro para enviar coordenadas de texturas.

# Request a buffer slot from GPU
buffer = glGenBuffers(3)


# ###  Enviando coordenadas de vértices para a GPU

vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)


# ###  Enviando coordenadas de textura para a GPU

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)


# dados de iluminacao
normals = np.zeros(len(normals_list), [("position", np.float32, 3)])
normals['position'] = normals_list

# upload coordenadas normais de cada vertice
glBindBuffer(GL_ARRAY_BUFFER, buffer[2])
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
stride = normals.strides[0]
offset = ctypes.c_void_p(0)
loc_normals_coord = glGetAttribLocation(program, "normals")
glEnableVertexAttribArray(loc_normals_coord)
glVertexAttribPointer(loc_normals_coord, 3, GL_FLOAT, False, stride, offset)

# posicao da fonte de luz
loc_light_pos = glGetUniformLocation(program, "lightPos")
glUniform3f(loc_light_pos, 0.0, 1.0, 0.0) #posicao da fonte de luz

# ### Desenhando nossos modelos
# * Cada modelo tem uma *Model* para posicioná-los no mundo.
# * É necessário saber qual a posição inicial e total de vértices de cada modelo.
# * É necessário indicar qual o `id` da textura do modelo.

def desenha_lago():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = -10.0; t_y = -1.0; t_z = -30.0;
    
    # escala
    s_x = 6.0; s_y = 6.0; s_z = 6.0; 
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

    ka = 0.3;
    kd = 0.5;

    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)
    
    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 0)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 0, 124*3) ## renderizando

def desenha_terreno():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = 0.0; t_y = -1.01; t_z = 0.0;
    
    # escala
    s_x = 40.0; s_y = 40.0; s_z = 40.0;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 1)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 372, 6) ## renderizando

def desenha_casa():
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # translacao
    t_x = 0.0; t_y = -1.0; t_z = 0.0;
    
    # escala
    s_x = 0.3; s_y = 0.3; s_z = 0.3;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 2)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 378, 594*3) ## renderizando

def desenha_barco(angle):
    # aplica a matriz model
    
    # rotacao
    r_x = 0.0; r_y = 1.0; r_z = 0.0;
    
    # translacao
    t_x = -10.0 + math.cos(angle)*4; t_y = -1.0; t_z = -30.0 + math.sin(angle)*4;
    
    # escala
    s_x = 0.3; s_y = 0.3; s_z = 0.3;
    
    mat_model = model(-angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 3)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 2160, 168*3) ## renderizando


def desenha_sol(t_x, t_y, t_z):
    # aplica a matriz model
    
    # rotacao
    angle = 0.0;
    r_x = 0.0; r_y = 0.0; r_z = 1.0;
    
    # escala
    s_x = 0.5; s_y = 0.5; s_z = 0.5;
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    loc_light_pos = glGetUniformLocation(program, "lightPos") # recuperando localizacao da variavel lightPos na GPU
    glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz

    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 4)
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 2664, 321*3) ## renderizando
    

# ### Eventos para modificar a posição da câmera.
# * Usar as teclas `A`, `S`, `D`, e `W` para movimentação no espaço tridimensional.
# * Usar a posição do mouse para "direcionar" a câmera.

cameraPos   = glm.vec3(0.0,  0.0,  1.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);


polygonal_mode = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode
    
    cameraSpeed = 0.2
    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 80 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 80 and action==1 and polygonal_mode==False:
            polygonal_mode=True
        
        
firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    
    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)


glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)


# ### Matrizes *Model*, *View*, e *Projection*
# Teremos uma aula específica para entender o seu funcionamento.

def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade
    
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando rotacao
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    
    matrix_transform = np.array(matrix_transform)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection


# ### Nesse momento, exibimos a janela.

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)


# ### Loop principal da janela.
# Enquanto a janela não for fechada, esse laço será executado. É neste espaço que trabalhamos com algumas interações com a `OpenGL`.

glEnable(GL_DEPTH_TEST) ### importante para 3D
   

rotacao_inc = 0
ang = 0
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)

    desenha_lago()   
    desenha_terreno()
    desenha_casa()
    
    rotacao_inc += 0.01
    desenha_barco(rotacao_inc)

    ang += 0.005
    desenha_sol(math.cos(ang)*30, math.sin(ang)*30, 0.0)
    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_TRUE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_TRUE, mat_projection)    
    
    glfw.swap_buffers(window)
glfw.terminate()
