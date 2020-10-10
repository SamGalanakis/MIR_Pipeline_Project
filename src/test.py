from pyvista import examples
import pyacvd

# download cow mesh

cow = examples.download_cow()

# plot original mesh
cow.plot(show_edges=True, color='w')


clus = pyacvd.Clustering(cow)
# mesh is not dense enough for uniform remeshing
clus.subdivide(3)
clus.cluster(20000)

# plot clustered cow mesh
clus.plot()


# remesh
remesh = clus.create_mesh()

# plot uniformly remeshed cow
remesh.plot(color='w', show_edges=True)