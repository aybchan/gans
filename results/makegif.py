# in command line run:
#    python makegif.py <dir of input pngs> <output filename> <fps>

if __name__ == '__main__':
    import sys
    import imageio
    import os

    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    if len(sys.argv) > 3:
        fps = sys.argv[3]
    else:
        fps = 2

    print(input_dir,output_filename)
    images = []

    for subdir, dirs, files in os.walk(input_dir):
        for file in sorted(files):
            print(file)
            file_path = os.path.join(subdir, file)
            if file_path.endswith('.png'):
                images.append(imageio.imread(file_path))

    imageio.mimsave(output_filename + '.gif', images,fps=fps)


