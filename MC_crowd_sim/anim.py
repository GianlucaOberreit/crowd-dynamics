import imageio
images = []

filenames = [f"crowd_simulation_results/states_N_300_hom/state_step_{i+1}.png" for i in range(300)]

for filename in filenames:
    images.append(imageio.imread_v2(filename))
imageio.mimsave(f"crowd_simulation_results/states_N_300_hom/states_anim.gif", images, fps = 25)


images = []

filenames = [f"crowd_simulation_results/states/state_step_{i+1}.png" for i in range(150)]

for filename in filenames:
    images.append(imageio.imread_v2(filename))
imageio.mimsave(f"crowd_simulation_results/states/states_anim.gif", images, fps = 25)

