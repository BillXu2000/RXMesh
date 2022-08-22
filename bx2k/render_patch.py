import pymeshlab, re, numpy as np, random

ml_ms = pymeshlab.MeshSet()
ml_ms.load_new_mesh('bunny0.obj')
ml_m = ml_ms.current_mesh()
vm = ml_m.vertex_matrix()
fm = ml_m.face_matrix()

with open("color.log", "r") as fi:
    lines = fi.readlines()

colors = np.zeros((fm.shape[0], 4))
# for i, j in enumerate(colors):
#     # colors[i] += (i % 64 + 1) / 64
#     # colors[i] = i / fm.shape[0]
#     for j in range(3):
#         colors[i][j] = random.random()
#     colors[i][3] = 1
patches = []
ma = 0
for line in lines:
    patch = [int(i) for i in re.findall('[0-9]+', line)]
    ma = max(ma, len(patch))
    patches.append(patch)
    color = [random.random() for i in range(3)]
    for f in patch:
        colors[f][:3] = color
        colors[f][3] = 1
print('max f in patch =', ma)
# print(patches)

# ml_ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vm, face_matrix=fm, v_color_matrix=vert_colors))
ml_ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vm, face_matrix=fm, f_color_matrix=colors))
ml_ms.save_current_mesh('out.obj')
