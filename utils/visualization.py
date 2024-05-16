import matplotlib.pyplot as plt
import numpy as np

def visual_3D(x,y,z, labels=None, showv=None, IsSave='', IsonlySave=False):
    """
    visualize 3D point clouds, with or without main direction
    :param x:       x
    :param y:       y
    :param z:       z
    :param labels:  labels of points
    :param showv:   [2*2*3] ([number of PCA principal components * pts_start_end(2) * xyz of pts_start_end]) array of eigenvectors (main directions of point clouds) for showing the main direction or not
    :param IsSave:
    :param IsonlySave: boolean
    :return:
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.scatter3D(x,y,z,s=3, c=z, cmap='jet') #cmap='viridis_r')

    if labels!=None:
        for li in range(len(labels)):
            ax.text(x[li], y[li], z[li]+0.01,labels[li], size=3)

    if showv is not None:
        if len(showv.shape)<3:
            showv = np.array([showv])
        for i in range(len(showv)):
            ax.plot(showv[i][:, 0], showv[i][:, 1], showv[i][:, 2], c='r') # x,y,z, color

    ### change the background into white
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    # plt.title(figtitle[0])

    # plt.suptitle(figtitle[0], fontsize=12)
    plt.tight_layout()

    if IsonlySave == False:
        plt.show()

    if IsSave!='':
        plt.savefig('./res/{}.png'.format(IsSave), dpi=300)
        plt.cla()
        plt.close(fig)


def visual_3D_withcolor(x,y,z, c=None, labels=None, showv=None, IsSave='', IsonlySave=False):
    """
    visualize 3D point clouds, with or without main direction
    :param x:       x
    :param y:       y
    :param z:       z
    :param c:       color
    :param labels:  labels of points
    :param showv:   [2*2*3] ([number of PCA principal components * pts_start_end(2) * xyz of pts_start_end]) array of eigenvectors (main directions of point clouds) for showing the main direction or not
    :param IsSave:
    :param IsonlySave: boolean
    :return:
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    if c is None:
        surf = ax.scatter3D(x,y,z,s=3, c=z, cmap='jet') #cmap='viridis_r')
    else:
        surf = ax.scatter3D(x, y, z, s=3, c=c, cmap='jet')  # cmap='viridis_r')

    if labels!=None:
        for li in range(len(labels)):
            ax.text(x[li], y[li], z[li]+0.01,labels[li], size=3)

    if showv is not None:
        if len(showv.shape)<3:
            showv = np.array([showv])
        for i in range(len(showv)):
            ax.plot(showv[i][:, 0], showv[i][:, 1], showv[i][:, 2], c='r') # x,y,z, color

    ### change the background into white
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    # plt.title(figtitle[0])

    # plt.suptitle(figtitle[0], fontsize=12)
    plt.tight_layout()

    if IsonlySave == False:
        plt.show()

    if IsSave!='':
        plt.savefig('./res/{}.png'.format(IsSave), dpi=300)
        plt.cla()
        plt.close(fig)


def visual_edges(x, y, z, faces, fin_corner_idx_pd, labels=None, IsSave='', IsonlySave=False):
    # draw
    color = ['black', 'brown', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'skyblue', 'blue', 'purple', 'violet']

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(x, y, z, s=7, cmap='jet')

    if labels!=None:
        for li in range(len(labels)):
            ax.text(x[li], y[li], z[li]+0.01,labels[li], size=3)

    for i in range(len(faces)):
        # draw
        facei = faces[i] + [faces[i][0]]
        line_starti = fin_corner_idx_pd.iloc[facei[:-1], 1:-1].to_numpy()  # ["id_x", "id_y", "id_z"]
        line_endi = fin_corner_idx_pd.iloc[facei[1:], 1:-1].to_numpy()

        line_segi = np.array([line_starti, line_endi])
        line_seg_xi = line_segi[:, :, 0].T  # [N_points, 2]
        line_seg_yi = line_segi[:, :, 1].T
        line_seg_zi = line_segi[:, :, 2].T

        for j in range(line_seg_xi.shape[0]):
            ax.plot(line_seg_xi[j], line_seg_yi[j], line_seg_zi[j], color=color[i%10])

    ### change the background into white
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('z', fontsize=10)
    # plt.title(figtitle[0])

    # plt.suptitle(figtitle[0], fontsize=12)
    plt.tight_layout()

    if IsonlySave == False:
        plt.show()

    if IsSave != '':
        plt.savefig('./res/{}.png'.format(IsSave), dpi=300)
        plt.cla()
        plt.close(fig)


def show_Dtri(vertices, dtri, all_pts):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.triplot(vertices[:, 0], vertices[:, 1], dtri)
    ax.plot(vertices[:, 0], vertices[:, 1], 'o')
    ax.scatter(all_pts[:, 0], all_pts[:, 1], c='orange', alpha=0.3)
    plt.show()


def visual_2D(x,y, labels=None, showv=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    surf = ax.scatter(x,y,s=1) #cmap='viridis_r')

    if showv is not None:
        if len(showv.shape)<3:
            showv = np.array([showv])
        for i in range(len(showv)):
            ax.plot(showv[i][:, 0], showv[i][:, 1], c='r') # x,y,z, color

    ### change the background into white
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    # ax.set_xlim((np.min(x)-10,np.max(x)+10))
    # ax.set_ylim((np.min(y)-10,np.max(y)+10))
    # ax.set_zlabel('z', fontsize=10)
    # plt.title(figtitle[0])

    # plt.suptitle(figtitle[0], fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.show()


def visual_2D_xyz(x,y,z, labels=None, showv=None):
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(131)
    ax1.scatter(x,y,s=3) #cmap='viridis_r')
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    plt.grid()

    ax1 = fig.add_subplot(132)
    ax1.scatter(y,z,s=3) #cmap='viridis_r')
    ax1.set_xlabel('y', fontsize=10)
    ax1.set_ylabel('z', fontsize=10)
    plt.grid()

    ax1 = fig.add_subplot(133)
    ax1.scatter(x,z,s=3) #cmap='viridis_r')
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('z', fontsize=10)
    plt.grid()

    plt.tight_layout()
    plt.show()